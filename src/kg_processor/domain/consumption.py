# SPDX-License-Identifier: Apache-2.0
"""What a graph consumed, and what that cost.

Consumption is recorded per graph so spend can be attributed to the work that
caused it, rather than inferred afterwards from a provider's invoice. Local runs
are recorded on the same terms as hosted ones: a local model bills nothing, but
the tokens it consumed are what make the avoided cost measurable, and a run whose
usage went unrecorded cannot be compared against anything.

Cost is derived here rather than stored by the adapters. Prices change, and a
recorded price becomes wrong silently; recorded *usage* stays true, so the rate
table can be corrected and every historical graph reprices correctly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class TokenUsage(BaseModel):
    """Tokens consumed by one provider call, in a shape every provider can report.

    Consumption is recorded for local models too. A local run bills nothing, but
    the tokens it consumed are what make the avoided cost measurable, so usage is
    a property of the work rather than of the invoice.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode="after")
    def _derive_total(self) -> TokenUsage:
        """Fill in a total the provider reported only as its two halves.

        Providers report ``total_tokens`` independently rather than as a sum, and
        several omit it while reporting prompt and completion counts. Consumers
        read the total, so leaving it at zero reports a billed call as free.
        """

        if not self.total_tokens and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )
        return self

    @classmethod
    def from_provider_payload(cls, payload: object) -> TokenUsage:
        """Read usage from a provider response, tolerating absence and naming drift.

        Providers disagree on names — ``input_tokens`` against ``prompt_tokens`` —
        and some omit usage entirely. A call whose usage cannot be read is recorded
        as zero rather than dropped, so the call still counts.
        """

        if not isinstance(payload, Mapping):
            return cls()
        prompt = _first_int(payload, ("prompt_tokens", "input_tokens"))
        completion = _first_int(payload, ("completion_tokens", "output_tokens"))
        total = _first_int(payload, ("total_tokens",)) or (prompt + completion)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def _first_int(payload: Mapping[str, Any], names: tuple[str, ...]) -> int:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
    return 0


class Locality(StrEnum):
    """Where the work ran, which decides whether it was billed or avoided."""

    HOSTED = "hosted"
    LOCAL = "local"


class ConsumptionEvent(BaseModel):
    """One measured unit of provider work.

    An event is emitted per provider call rather than per stage so that a stage
    which fans out across windows reports what each window actually consumed.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    graph_id: str
    run_id: str = ""
    task_id: str = ""
    task_attempt: int = 0
    job_id: str = ""
    stage: str
    operation: str
    provider: str
    model: str
    locality: Locality = Locality.HOSTED
    calls: int = 1
    usage: TokenUsage = Field(default_factory=TokenUsage)
    pages: int = 0
    file_id: str | None = None


class Rate(BaseModel):
    """The price of one provider's model.

    Token rates are quoted per million tokens because that is how providers
    publish them; quoting per token invites silent factor-of-a-million errors.

    Two units, because providers bill in two: an hourly API quotes dollars, while
    Snowflake quotes AI credits and the dollar value of a credit is a property of
    the account's contract rather than of the model. Credits are converted with
    :attr:`RateCard.usd_per_credit`, so correcting one number reprices every
    Snowflake model at once instead of thirty hand-multiplied figures drifting
    apart.
    """

    prompt_usd_per_million: float = 0.0
    completion_usd_per_million: float = 0.0
    usd_per_page: float = 0.0
    prompt_credits_per_million: float = 0.0
    completion_credits_per_million: float = 0.0
    credits_per_page: float = 0.0

    def usd(self, *, usd_per_credit: float) -> tuple[float, float, float]:
        """Return this rate as dollars for prompt, completion and page units."""

        return (
            self.prompt_usd_per_million + self.prompt_credits_per_million * usd_per_credit,
            self.completion_usd_per_million + self.completion_credits_per_million * usd_per_credit,
            self.usd_per_page + self.credits_per_page * usd_per_credit,
        )


class RateCard(BaseModel):
    """Prices by provider and model, plus what local work would have cost.

    ``local_reference`` names the hosted model a local run is priced against. A
    local run has no invoice, so its saving is only meaningful relative to a
    stated alternative — without one, "saved" would be an unfalsifiable number.

    ``usd_per_credit`` is what one Snowflake AI credit costs this account. It has
    no default worth guessing: the figure comes from the customer's contract, and
    inventing one would report a confident price that is quietly wrong.
    """

    rates: dict[str, Rate] = Field(default_factory=dict)
    local_reference: str | None = None
    usd_per_credit: float = 0.0

    def rate_for(self, provider: str, model: str) -> Rate | None:
        """Find the most specific rate configured for a call.

        A provider-wide entry is a deliberate fallback: a new model should be
        priced approximately rather than silently counted as free.
        """

        for key in (f"{provider}:{model}", f"{provider}:*", model):
            rate = self.rates.get(key)
            if rate is not None:
                return rate
        return None


class ConsumptionTotals(BaseModel):
    """Aggregated consumption for one grouping, with cost and avoided cost.

    ``unpriced_calls`` is reported rather than hidden. A total that quietly omits
    calls it had no rate for reads as authoritative while being wrong, so the gap
    is made visible and the operator decides whether it matters.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    pages: int = 0
    billed_usd: float = 0.0
    avoided_usd: float = 0.0
    unpriced_calls: int = 0

    @property
    def is_complete(self) -> bool:
        """Whether every call in this total could be priced."""

        return self.unpriced_calls == 0


def price_event(event: ConsumptionEvent, card: RateCard) -> tuple[float, bool]:
    """Return the cost of one event, and whether it was fully priced.

    A rate only counts as covering an event if it prices the dimension the event
    actually used. A card that prices tokens but not pages would otherwise report
    parsing a thousand pages as costing nothing — the same silent zero that
    reporting unpriced calls exists to prevent.
    """

    rate = card.rate_for(event.provider, event.model)
    if rate is None:
        return 0.0, False
    prompt_usd, completion_usd, page_usd = rate.usd(usd_per_credit=card.usd_per_credit)
    tokens = (
        event.usage.prompt_tokens * prompt_usd + event.usage.completion_tokens * completion_usd
    ) / 1_000_000
    used_tokens = event.usage.prompt_tokens or event.usage.completion_tokens
    covers_tokens = not used_tokens or bool(prompt_usd or completion_usd)
    covers_pages = not event.pages or bool(page_usd)
    return tokens + event.pages * page_usd, covers_tokens and covers_pages


def summarize(events: Iterable[ConsumptionEvent], card: RateCard) -> ConsumptionTotals:
    """Aggregate events, separating what was billed from what was avoided.

    Hosted work accrues billed cost. Local work accrues avoided cost, priced at
    the configured reference model — the question being answered is not "what did
    this cost" but "what would it have cost had we not run it ourselves".
    """

    totals = ConsumptionTotals()
    for event in events:
        totals.calls += event.calls
        totals.prompt_tokens += event.usage.prompt_tokens
        totals.completion_tokens += event.usage.completion_tokens
        totals.total_tokens += event.usage.total_tokens
        totals.pages += event.pages
        if event.locality is Locality.LOCAL:
            avoided, priced = _price_as_reference(event, card)
            totals.avoided_usd += avoided
        else:
            billed, priced = price_event(event, card)
            totals.billed_usd += billed
        if not priced:
            totals.unpriced_calls += event.calls
    return totals


def _price_as_reference(event: ConsumptionEvent, card: RateCard) -> tuple[float, bool]:
    """Price local work at the hosted model it stands in for."""

    if not card.local_reference:
        return 0.0, False
    provider, _, model = card.local_reference.partition(":")
    return price_event(
        event.model_copy(update={"provider": provider, "model": model or provider}),
        card,
    )


def group_by(
    events: Iterable[ConsumptionEvent],
    key: str,
    card: RateCard,
) -> dict[str, ConsumptionTotals]:
    """Aggregate events by one of their own attributes."""

    grouped: dict[str, list[ConsumptionEvent]] = {}
    for event in events:
        grouped.setdefault(str(getattr(event, key, "") or "unknown"), []).append(event)
    return {name: summarize(items, card) for name, items in sorted(grouped.items())}
