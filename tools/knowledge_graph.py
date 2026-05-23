import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

class CodeKnowledgeGraph:
    """
    A lightweight, pure-Python, dependency-free knowledge graph.
    Designed for representation of codebases, structural dependencies, and semantic links.
    Fully serializable to standard JSON files.
    """
    def __init__(self):
        # Dictionary structure:
        # node_id -> {
        #    "id": str,
        #    "type": str,          # 'file', 'class', 'function', 'dependency', 'concept'
        #    "properties": dict,   # e.g., signature, line numbers, docstring, etc.
        #    "edges": {            # maps edge_type -> list of target node_ids
        #         "edge_type": [target_id1, target_id2, ...]
        #    }
        # }
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node_id: str, node_type: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Adds a node to the knowledge graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "properties": properties or {},
                "edges": {}
            }
        else:
            # Update properties if already exists
            if properties:
                self.nodes[node_id]["properties"].update(properties)

    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Creates a directed edge of a specific type from source_id to target_id."""
        # Ensure both nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        edges = self.nodes[source_id]["edges"]
        if edge_type not in edges:
            edges[edge_type] = []

        if target_id not in edges[edge_type]:
            edges[edge_type].append(target_id)
        return True

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Returns outgoing neighbors of a node. Can be filtered by edge_type."""
        if node_id not in self.nodes:
            return []

        node_data = self.nodes[node_id]
        if edge_type:
            return node_data["edges"].get(edge_type, [])

        # Return all targets across all edge types
        neighbors = []
        for targets in node_data["edges"].values():
            neighbors.extend(targets)
        return list(set(neighbors))

    def get_incoming_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Returns nodes that point TO this node. Can be filtered by edge_type."""
        incoming = []
        for source_id, node_data in self.nodes.items():
            if edge_type:
                targets = node_data["edges"].get(edge_type, [])
                if node_id in targets:
                    incoming.append(source_id)
            else:
                for targets in node_data["edges"].values():
                    if node_id in targets:
                        incoming.append(source_id)
                        break
        return incoming

    def traverse(self, start_node: str, max_depth: int = 2, edge_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs a Breadth-First Search (BFS) traversal up to max_depth.
        Returns a subgraph representation containing all visited nodes and traversed edges.
        """
        if start_node not in self.nodes:
            return {"nodes": {}, "edges": []}

        visited_nodes = {}
        visited_edges = []
        
        # Queue stores: (node_id, current_depth)
        queue = [(start_node, 0)]
        visited_set = {start_node}

        while queue:
            current_id, depth = queue.pop(0)
            node_data = self.nodes[current_id]
            visited_nodes[current_id] = {
                "id": current_id,
                "type": node_data["type"],
                "properties": node_data["properties"]
            }

            if depth >= max_depth:
                continue

            # Process outgoing edges
            for e_type, targets in node_data["edges"].items():
                if edge_type and e_type != edge_type:
                    continue
                for target in targets:
                    visited_edges.append({
                        "source": current_id,
                        "target": target,
                        "type": e_type
                    })
                    if target not in visited_set:
                        visited_set.add(target)
                        queue.append((target, depth + 1))

        return {
            "nodes": visited_nodes,
            "edges": visited_edges
        }

    def find_shortest_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Finds the shortest path from start_node to end_node using BFS. Returns list of node IDs."""
        if start_node not in self.nodes or end_node not in self.nodes:
            return None

        if start_node == end_node:
            return [start_node]

        queue = [[start_node]]
        visited = {start_node}

        while queue:
            path = queue.pop(0)
            current = path[-1]

            for neighbor in self.get_neighbors(current):
                if neighbor == end_node:
                    return path + [end_node]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def save_to_disk(self, file_path: str) -> str:
        """Saves the graph to a JSON file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.nodes, f, indent=2, ensure_ascii=False)
            return f"Successfully saved knowledge graph to {file_path}"
        except Exception as e:
            return f"Error saving graph to disk: {str(e)}"

    def load_from_disk(self, file_path: str) -> bool:
        """Loads graph data from a JSON file. Returns True on success."""
        path = Path(file_path)
        if not path.exists():
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            return True
        except Exception:
            return False

    def trace_symbol(self, symbol: str) -> str:
        """Looks up a symbol in the graph and returns a detailed structural summary of imports and calls."""
        # Find matching nodes
        matches = []
        for n_id, n_data in self.nodes.items():
            name = n_data.get("properties", {}).get("name", "")
            if name.lower() == symbol.lower() or symbol.lower() in n_id.lower():
                matches.append((n_id, n_data))
        
        if not matches:
            return f"Symbol '{symbol}' not found in the Knowledge Graph."
            
        results = []
        for n_id, n_data in matches[:3]: # Limit to top 3
            n_type = n_data["type"]
            props = n_data.get("properties", {})
            name = props.get("name", n_id)
            path = props.get("path", n_id.split(':')[1] if ':' in n_id else "")
            start_line = props.get("start_line", "?")
            
            lines = [f"### Symbol: {name} ({n_type.upper()})"]
            lines.append(f"  Defined in: {path} (Line {start_line})")
            
            outgoing = []
            for e_type, targets in n_data.get("edges", {}).items():
                for t in targets:
                    t_name = self.nodes.get(t, {}).get("properties", {}).get("name", t)
                    outgoing.append(f"    - {e_type} -> {t_name}")
                    
            incoming = []
            for s_id, s_data in self.nodes.items():
                for e_type, targets in s_data.get("edges", {}).items():
                    if n_id in targets:
                        s_name = s_data.get("properties", {}).get("name", s_id)
                        incoming.append(f"    - {s_name} -> {e_type}")
                        
            if outgoing:
                lines.append("  Outward Relations:")
                lines.extend(outgoing[:8])
            else:
                lines.append("  No outgoing relations.")
                
            if incoming:
                lines.append("  Inward Relations:")
                lines.extend(incoming[:8])
            else:
                lines.append("  No incoming relations.")
            results.append("\n".join(lines))
            
        return "\n\n".join(results)

    def impact_analysis(self, target: str) -> str:
        """Calculates the change impact blast radius for a given target symbol."""
        # Find matching node
        target_node_id = None
        for n_id, n_data in self.nodes.items():
            name = n_data.get("properties", {}).get("name", "")
            path = n_data.get("properties", {}).get("path", "")
            if target.lower() in n_id.lower() or target.lower() == name.lower() or target.lower() in path.lower():
                target_node_id = n_id
                break
                
        if not target_node_id:
            return f"Target '{target}' not found in Knowledge Graph."
            
        queue = [(target_node_id, 0)]
        visited = {target_node_id}
        impacted_by_distance = {}
        
        while queue:
            curr_id, dist = queue.pop(0)
            if dist > 0:
                if dist not in impacted_by_distance:
                    impacted_by_distance[dist] = []
                impacted_by_distance[dist].append(self.nodes[curr_id])
                
            for source_id, s_data in self.nodes.items():
                if source_id in visited:
                    continue
                for edge_type, targets in s_data.get("edges", {}).items():
                    if curr_id in targets:
                        visited.add(source_id)
                        queue.append((source_id, dist + 1))
                        break
                        
        t_data = self.nodes[target_node_id]
        t_name = t_data.get("properties", {}).get("name", target_node_id)
        t_type = t_data["type"]
        
        lines = [f"### Impact Footprint for {t_name} ({t_type.upper()})"]
        if not impacted_by_distance:
            lines.append("Isolated Component: Safe to modify with zero downstream structural impact.")
            return "\n".join(lines)
            
        lines.append(f"Modifying this component has a blast radius of {len(visited) - 1} dependents:")
        for dist in sorted(impacted_by_distance.keys()):
            lines.append(f"  Distance {dist} (Blast Zone):")
            for dep in impacted_by_distance[dist]:
                d_name = dep.get("properties", {}).get("name", dep["id"])
                d_type = dep["type"]
                d_path = dep.get("properties", {}).get("path", "")
                lines.append(f"    • {d_name} ({d_type}) in {d_path}")
        return "\n".join(lines)
