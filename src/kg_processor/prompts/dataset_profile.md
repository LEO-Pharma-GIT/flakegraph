You profile a spreadsheet or other tabular data file for a knowledge graph.
From its file name and opening rows, decide what it is:
catalogue - each row names a thing of one of the listed entity types, such as a
material, product, substance, or organization, that the graph should contain;
measurements - rows or columns record values, limits, or results for named things
of the listed types;
other - calculations, templates, schedules, plans, forms, administrative or
free-form sheets whose rows are not things of the listed types.
Then summarize in one or two sentences what the file contains, naming the things
it is about as the file names them. Do not guess beyond what the rows show.
For a catalogue, also map its columns from the header row, one entry per column,
naming each column exactly as its header writes it:
subject - the column whose cells name the thing each row is about, with its
entity type; exactly one column is the subject;
relation - its cells name another thing, of an entity type, that one of the
listed relation types links to the row's subject, such as its supplier, maker,
source, category it belongs to, or a thing it is used in; give the entity type
and the relation type, choosing one whose source and target types admit the two;
value - its cells are values, amounts, limits, or grades; when a relation type
that carries quantities states the value together with a relation column, as a
maximum amount is stated for the thing it is used in, give that column's header
in of_column, and otherwise leave of_column empty;
alias - its cells are a code, number, or other identifier of the subject itself;
none - anything else, such as remarks, dates of entry, or row numbers.
Leave entity_type and relation_type empty where a role does not use them. Use
only the listed types and relations. For measurements and other, return no
columns.
