You read one picture taken from a document so that its content can be searched and quoted as text.
Say what the picture shows, using only what is visible in it. The page context names the document and page the picture sits on; use it only to understand labels the picture abbreviates, never as a source of facts the picture does not show.

Choose kind:
chart - a plotted chart or graph (bars, lines, pie, scatter, area);
table - a grid of rows and columns of values;
diagram - a flow, process, map, timeline, org chart or other drawing that relates things;
photo - a photograph of people, places, products or objects;
screenshot - an image of a screen, a web page, an app or another slide;
text - mostly text, such as a scanned paragraph, a quote or a callout;
decoration - a logo, icon, divider, background or other picture that states nothing;
other - anything else.

Write description as one to three plain sentences on what the picture shows and what it says: the subject, and for a chart what is plotted against what and the main pattern. Write text as the words and numbers legible in the picture, in reading order, one line per line of text; leave it empty when there are none.

For a chart, fill charts with what it plots: chart_type, title, the axis titles, the categories in order, and each series with its name and one value per category, as numbers in the units the chart shows (a percent as its number, 42 for 42%). Read values from data labels where the chart prints them; where it does not, estimate from the axis and gridlines. Use null for a value you cannot read. Return several charts only when the picture holds several.
For a table, fill tables with its rows, header rows first, each cell's text as written, and say how many leading rows are headers.
Leave charts and tables empty for every other kind, and return a decoration with an empty description and text.
Never invent names, numbers or labels that are not in the picture.
