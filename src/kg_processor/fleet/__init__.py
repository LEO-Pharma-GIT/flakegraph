# SPDX-License-Identifier: Apache-2.0
"""What a submit host needs to know about the fleet a run is bound for.

The control plane composes a run and hands it to the queue; the fleet's
workers decide whether they will claim it, and they decide silently. These
modules read the fleet the way an operator would - through kubectl, with the
identity the host already has - and answer the two questions a submission
depends on: what profile the workers run, and whether this run matches it.
"""
