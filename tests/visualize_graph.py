"""Export the Kuzu graph to an interactive HTML visualization using vis.js."""

from __future__ import annotations

import json
import os
import webbrowser

import kuzu

DB_PATH = "./explore.db"
OUT_PATH = "./graph_viz.html"

COLORS = {
    "PERSON":   "#4A90D9",
    "ORG":      "#E67E22",
    "CONCEPT":  "#27AE60",
    "LOCATION": "#8E44AD",
    "EVENT":    "#E74C3C",
    "Document": "#95A5A6",
    "Chunk":    "#BDC3C7",
}


def main() -> None:
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)

    # Collect nodes
    nodes = {}

    for label, color in [("Document", COLORS["Document"]), ("Chunk", COLORS["Chunk"])]:
        try:
            r = conn.execute(f"MATCH (n:{label}) RETURN n.id, n.title")
            while r.has_next():
                row = r.get_next()
                nid, title = row[0], row[1] or row[0]
                nodes[nid] = {"id": nid, "label": title[:30], "color": color, "group": label}
        except Exception:
            pass

    r = conn.execute("MATCH (e:Entity) RETURN e.id, e.name, e.type")
    while r.has_next():
        row = r.get_next()
        eid, name, etype = row[0], row[1], row[2]
        nodes[eid] = {"id": eid, "label": name, "color": COLORS.get(etype, "#AAB7B8"), "group": etype}

    # Collect edges
    edges = []
    edge_id = 0

    try:
        r = conn.execute("MATCH (d:Document)-[:CONTAINS]->(c:Chunk) RETURN d.id, c.id")
        while r.has_next():
            row = r.get_next()
            edges.append({"id": edge_id, "from": row[0], "to": row[1], "label": "CONTAINS", "arrows": "to"})
            edge_id += 1
    except Exception:
        pass

    try:
        r = conn.execute("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c.id, e.id")
        while r.has_next():
            row = r.get_next()
            edges.append({"id": edge_id, "from": row[0], "to": row[1], "label": "MENTIONS", "arrows": "to"})
            edge_id += 1
    except Exception:
        pass

    r = conn.execute("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.id, b.id, r.type")
    while r.has_next():
        row = r.get_next()
        edges.append({"id": edge_id, "from": row[0], "to": row[1], "label": row[2], "arrows": "to"})
        edge_id += 1

    nodes_json = json.dumps(list(nodes.values()))
    edges_json = json.dumps(edges)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>GraphRAG Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; background: #1a1a2e; font-family: Arial, sans-serif; }}
    #graph {{ width: 100vw; height: 100vh; }}
    #legend {{ position: absolute; top: 16px; left: 16px; background: rgba(0,0,0,0.7);
               color: white; padding: 12px 16px; border-radius: 8px; font-size: 13px; }}
    #legend h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px;
            border-radius: 50%; margin-right: 6px; }}
    #stats {{ position: absolute; top: 16px; right: 16px; background: rgba(0,0,0,0.7);
              color: white; padding: 12px 16px; border-radius: 8px; font-size: 13px; }}
  </style>
</head>
<body>
  <div id="graph"></div>
  <div id="legend">
    <h3>Node Types</h3>
    {''.join(f'<div><span class="dot" style="background:{c}"></span>{t}</div>' for t, c in COLORS.items())}
  </div>
  <div id="stats">
    <b>{len(nodes)}</b> nodes &nbsp;|&nbsp; <b>{len(edges)}</b> edges
  </div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById('graph');
    const network = new vis.Network(container, {{ nodes, edges }}, {{
      background: '#1a1a2e',
      nodes: {{ font: {{ color: '#ffffff', size: 13 }}, borderWidth: 2, shadow: true }},
      edges: {{ font: {{ color: '#cccccc', size: 11, align: 'middle' }},
               color: {{ color: '#555577' }}, smooth: {{ type: 'curvedCW', roundness: 0.2 }} }},
      physics: {{ solver: 'forceAtlas2Based', stabilization: {{ iterations: 150 }} }},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }});
  </script>
</body>
</html>"""

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Graph exported: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Opening {OUT_PATH} in browser...")
    webbrowser.open(os.path.abspath(OUT_PATH))


if __name__ == "__main__":
    main()
