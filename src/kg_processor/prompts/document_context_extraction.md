Identify focal entities needed to interpret document-wide discourse.
Use only entity types explicitly enabled for document context, and follow each
type's description, guidance, and counter_examples.
When a type for the source itself - a paper, report, or other work - is enabled
and the front matter explicitly identifies this source's own title, return
exactly one record of it first; do not omit it in favour of another focal entity.
Then independently audit every supplied title, abstract, summary, and
front-matter contribution sentence for the primary named things the source
presents as its own work, such as a method, model, product, or system; the first
record never makes this audit complete. Include an acronym expansion or a
descriptive name when the source explicitly identifies it as its own.
For a certificate, form, label, data sheet, or specification, the focal entity is
the named product, material, substance, or batch its header describes, when that
type is enabled; for another document, the named report, case, standard, or
product whose identity its front matter establishes.
Do not return cited works, background or comparison things, class terms, or
section headings.
A location page (page 0) names the folders the document sits in and its file.
The file is this document: its file name and folder paths are not entities. A
folder or file name may still name the document's subject, such as the product,
material, batch, project, or experiment it belongs to.
Return only explicit identities grounded by a short verbatim front-matter quote.
The quote must contain the canonical name or a genuine spelling/number alias.
Do not return authors, affiliations, references, topics, methods, or datasets unless
one of those types is itself enabled for document context.
Use each type's supplied contextual surfaces only as later discourse guidance.
Return an empty list when no configured focal entity is explicit.
