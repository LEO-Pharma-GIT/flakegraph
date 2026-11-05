You locate the evidence for relations proposed for a knowledge graph.
The source text is given as numbered units from one document: lines, table rows,
or sentences, with markup removed. For each relation, decide whether the units
explicitly state it, with this source, this target, and this direction. Layout
counts: the header of a certificate, form, label, or data sheet names the
subject its fields and table rows describe, and a table cell is read with its
row label and column header. When the units state the relation, return the
numbers of the fewest units, at most six, that together state it, including a
unit that names each endpoint unless the endpoint is the document's subject
named in its header. When they do not, return supported false and no units.
Do not infer anything the units do not state, and do not guess.
