# SPDX-License-Identifier: Apache-2.0
"""What a parser can tell about a document's pictures without looking at them.

Parsing records each picture as an asset: where it lives in the file, what
kind of image it is, a hash of its bytes and its size in pixels. A later step
that reads pictures (written separately) fetches the bytes through the
parser's ``read_image_bytes``. Nothing here decodes pixels during parsing:
Pillow reads only an image's header for its size, and a PDF image's size is
in its dictionary.

PDF images are found by walking each page's XObject resources, through form
XObjects, so the parse does not decode any image stream. Only when a picture
is read is its stream decoded and, unless it is already a JPEG, written out
as a PNG, the two encodings every picture reader takes.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError
from pypdf import PageObject
from pypdf.generic import ArrayObject, DictionaryObject, StreamObject

# PDF images smaller than this on either side, or with fewer pixels in all,
# are bullets, rules, icons and logo fragments rather than pictures with
# something to read; a picture reader would spend a request on each of them.
MIN_PDF_IMAGE_SIDE_PX = 64
MIN_PDF_IMAGE_PIXELS = 128 * 128
# Form XObjects nest; a hostile file can nest them without end or in a loop.
MAX_PDF_FORM_DEPTH = 8
JPEG_CONTENT_TYPE = "image/jpeg"
PNG_CONTENT_TYPE = "image/png"
# The content types of picture parts Office packages commonly hold, by file
# extension, for packages whose [Content_Types].xml does not say.
_EXTENSION_CONTENT_TYPES = {
    "png": PNG_CONTENT_TYPE,
    "jpg": JPEG_CONTENT_TYPE,
    "jpeg": JPEG_CONTENT_TYPE,
    "jpe": JPEG_CONTENT_TYPE,
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "emf": "image/x-emf",
    "wmf": "image/x-wmf",
    "svg": "image/svg+xml",
    "wdp": "image/vnd.ms-photo",
    "jxr": "image/vnd.ms-photo",
    "webp": "image/webp",
}
_JPEG_FILTERS = frozenset({"/DCTDecode", "/DCT"})


@dataclass(frozen=True)
class ImageFacts:
    """A picture's hash, byte size and, when its header says, pixel size."""

    sha256: str
    size_bytes: int
    width_px: int | None
    height_px: int | None


def image_facts(payload: bytes) -> ImageFacts:
    """Hash a picture's bytes and read its pixel size from its header.

    Pillow opens an image lazily, reading the header only. A header Pillow
    does not know (EMF, SVG) or that claims an absurd size leaves the pixel
    size unknown rather than failing the document.
    """

    width: int | None = None
    height: int | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        width = height = None
    return ImageFacts(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        width_px=width,
        height_px=height,
    )


def content_type_for_member(member: str, declared: dict[str, str]) -> str | None:
    """A package part's content type: as declared, else by its extension.

    ``declared`` maps part names (``/ppt/media/image1.png``) and lower-case
    extensions to the types ``[Content_Types].xml`` gives them.
    """

    override = declared.get("/" + member.lstrip("/"))
    if override:
        return override
    extension = PurePosixPath(member).suffix.lower().lstrip(".")
    return declared.get(extension) or _EXTENSION_CONTENT_TYPES.get(extension)


@dataclass(frozen=True)
class PdfImageRef:
    """An image XObject a PDF page draws, found without decoding it.

    ``path`` is the chain of XObject names from the page's resources to the
    image, through any form XObjects that hold it.
    """

    path: tuple[str, ...]
    width_px: int
    height_px: int
    content_type: str
    sha256: str
    size_bytes: int


def pdf_page_images(page: PageObject) -> Iterator[PdfImageRef]:
    """The images a page's resources hold that are large enough to read.

    Each distinct image object is yielded once per page. Stencil masks
    (``/ImageMask``) are shapes, not pictures, and are skipped. The hash is
    of the stream as stored, so the same picture placed on many pages, or
    in many files, hashes alike.
    """

    seen: set[tuple[int, int] | int] = set()
    yield from _walk_xobjects(page, (), seen, depth=0)


def pdf_image_bytes(page: PageObject, path: tuple[str, ...]) -> tuple[bytes, str]:
    """Decode one image a page draws, as JPEG or PNG bytes and their type.

    A JPEG stream is returned as stored. Any other image is decoded by pypdf
    and written as a PNG; colour spaces PNG cannot hold (CMYK) are converted
    to RGB. An image larger than Pillow's decompression-bomb bound is
    refused before it is decoded.
    """

    image = _resolve_image(page, path)
    width, height = _int(image.get("/Width")), _int(image.get("/Height"))
    if width * height > (Image.MAX_IMAGE_PIXELS or width * height):
        raise ValueError(f"PDF image of {width}x{height} pixels exceeds the decoding bound")
    if _is_jpeg(image):
        return image.get_data(), JPEG_CONTENT_TYPE
    key: str | tuple[str, ...] = path[0] if len(path) == 1 else path
    try:
        decoded = page.images[key].image  # type: ignore[index]
    except (Image.DecompressionBombError, UnidentifiedImageError) as exc:
        raise ValueError(f"PDF image {'/'.join(path)} cannot be decoded") from exc
    if decoded is None:
        raise ValueError(f"PDF image {'/'.join(path)} cannot be decoded")
    if decoded.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}:
        decoded = decoded.convert("RGB")
    buffer = io.BytesIO()
    decoded.save(buffer, format="PNG")
    return buffer.getvalue(), PNG_CONTENT_TYPE


def _walk_xobjects(
    holder: DictionaryObject,
    path: tuple[str, ...],
    seen: set[tuple[int, int] | int],
    *,
    depth: int,
) -> Iterator[PdfImageRef]:
    if depth > MAX_PDF_FORM_DEPTH:
        return
    resources = _dictionary(holder.get("/Resources"))
    xobjects = _dictionary(resources.get("/XObject")) if resources is not None else None
    if xobjects is None:
        return
    for name in list(xobjects):
        reference = xobjects.raw_get(name)
        identity: tuple[int, int] | int = (
            (reference.idnum, reference.generation)
            if hasattr(reference, "idnum")
            else id(reference)
        )
        if identity in seen:
            continue
        seen.add(identity)
        entry = xobjects[name]
        if not isinstance(entry, StreamObject):
            continue
        subtype = entry.get("/Subtype")
        if subtype == "/Form":
            yield from _walk_xobjects(entry, (*path, str(name)), seen, depth=depth + 1)
        elif subtype == "/Image":
            found = _image_ref(entry, (*path, str(name)))
            if found is not None:
                yield found


def _image_ref(image: StreamObject, path: tuple[str, ...]) -> PdfImageRef | None:
    if image.get("/ImageMask") in {True, "true"}:
        return None
    width, height = _int(image.get("/Width")), _int(image.get("/Height"))
    if min(width, height) < MIN_PDF_IMAGE_SIDE_PX or width * height < MIN_PDF_IMAGE_PIXELS:
        return None
    stored = _stored_bytes(image)
    return PdfImageRef(
        path=path,
        width_px=width,
        height_px=height,
        content_type=JPEG_CONTENT_TYPE if _is_jpeg(image) else PNG_CONTENT_TYPE,
        sha256=hashlib.sha256(stored).hexdigest(),
        size_bytes=len(stored),
    )


def _stored_bytes(stream: StreamObject) -> bytes:
    """The stream's bytes as the file stores them, still encoded.

    pypdf keeps the encoded bytes of a stream it read in ``_data``; reading
    them avoids decompressing every image only to hash it.
    """

    stored: Any = getattr(stream, "_data", None)
    return bytes(stored) if isinstance(stored, bytes | bytearray) else stream.get_data()


def _resolve_image(page: PageObject, path: tuple[str, ...]) -> StreamObject:
    holder: DictionaryObject = page
    for depth, name in enumerate(path):
        if depth > MAX_PDF_FORM_DEPTH:
            break
        resources = _dictionary(holder.get("/Resources"))
        xobjects = _dictionary(resources.get("/XObject")) if resources is not None else None
        entry = xobjects.get(name) if xobjects is not None else None
        entry = entry.get_object() if entry is not None else None
        if not isinstance(entry, StreamObject):
            break
        if depth == len(path) - 1:
            if entry.get("/Subtype") == "/Image":
                return entry
            break
        holder = entry
    raise ValueError(f"PDF page has no image at {'/'.join(path)}")


def _is_jpeg(image: StreamObject) -> bool:
    filters = image.get("/Filter")
    if isinstance(filters, ArrayObject):
        names = [str(item) for item in filters]
        return bool(names) and names[-1] in _JPEG_FILTERS
    return str(filters) in _JPEG_FILTERS if filters is not None else False


def _dictionary(value: object) -> DictionaryObject | None:
    resolved = value.get_object() if hasattr(value, "get_object") else value
    return resolved if isinstance(resolved, DictionaryObject) else None


def _int(value: object) -> int:
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
