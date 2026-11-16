# Deep Learning Papers Dataset Terms

The files in this directory that were created for FlakeGraph, including the
gold annotations, ontology, benchmark metadata, downloader, and documentation,
are provided under the repository's Apache License 2.0 unless a file states
otherwise.

The research papers are third-party works and are not distributed by this
repository. Downloading a paper does not change its copyright or license. Users
are responsible for determining whether their intended use is permitted by the
paper's authors, publisher, and applicable law.

`gold.json` quotes each identified paper only as evidence locators: the
`evidence_contains` anchors and the `sentence` field, which carries the same
anchor, are phrases of at most ten words each, and a few entity aliases are
short phrases of the same length taken from a paper. No longer passage is
reproduced, and a contract test enforces the limit. These short phrases are
excluded from the Apache License grant; their copyright and permitted uses
remain governed by their respective sources and applicable law. The Apache
License applies to the original annotation structure, canonical labels,
configuration, code, and documentation.

The Bau Lab reading list is used only as a public index of source locations.
FlakeGraph is not affiliated with Bau Lab or the listed authors and publishers.
