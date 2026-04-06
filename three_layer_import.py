"""
Three-layer architecture import script
Full implementation of the paper's three-layer architecture: Bottom/Middle/Top
"""

import os
import sys
import argparse
from pathlib import Path

from camel.storages import Neo4jGraph
from simple_neo4j_graph import SimpleNeo4jGraph
from dataloader import load_high
from creat_graph import creat_metagraph
from utils import str_uuid, ref_link


class ThreeLayerImporter:
    """Three-layer architecture importer"""
    
    def __init__(self, neo4j_url, neo4j_username, neo4j_password):
        """Initialize"""
        print("\n" + "="*80)
        print("Three-Layer Medical Knowledge Graph Importer")
        print("="*80)
        
        # Connect to Neo4j
        print("\n[Connecting to Neo4j]...")
        try:
            self.n4j = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password
            )
            print("✅ Neo4j connected successfully")
        except Exception as e:
            print(f"Neo4jGraph init failed ({e}). Falling back to SimpleNeo4jGraph without APOC.")
            self.n4j = SimpleNeo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password
            )
            print("✅ SimpleNeo4jGraph connected successfully")
        
        # Store GIDs for each layer
        self.layer_gids = {
            'bottom': [],
            'middle': [],
            'top': []
        }
    
    def clear_database(self):
        """Clear the database (no confirmation required)"""
        print("\n[Clearing Database]...")
        result = self.n4j.query("MATCH (n) RETURN count(n) as count")
        count = result[0]['count'] if result else 0
        print(f"Current node count: {count}")
        
        if count > 0:
            print("Clearing database...")
            self.n4j.query("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared")
        else:
            print("Database is already empty")
    
    def import_layer(self, layer_name: str, data_path: str, args):
        """
        Import data for one layer
        
        Args:
            layer_name: Layer name (bottom/middle/top)
            data_path: Data path
            args: Additional arguments
        """
        print("\n" + "="*80)
        print(f"[{layer_name.upper()} LAYER] Starting import")
        print(f"Data path: {data_path}")
        print("="*80)
        
        data_path = Path(data_path)
        
        # Get all text files
        if data_path.is_file():
            files = [data_path]
        else:
            files = list(data_path.glob("*.txt"))
            # Recursively find txt files in subdirectories
            files.extend(data_path.rglob("*/*.txt"))
        
        print(f"\nFound {len(files)} file(s)")
        
        # Process each file
        for idx, file_path in enumerate(files, 1):
            print(f"\n{'─'*80}")
            print(f"[File {idx}/{len(files)}] {file_path.name}")
            print(f"{'─'*80}")
            
            try:
                # Read content
                content = load_high(str(file_path))
                
                if not content or len(content.strip()) < 50:
                    print(f"⚠️  Skipping: content too short")
                    continue
                
                # Generate GID
                gid = str_uuid()
                self.layer_gids[layer_name].append(gid)
                
                # Build graph
                self.n4j = creat_metagraph(args, content, gid, self.n4j)
                
                print(f"✅ Done: {file_path.name} (GID: {gid[:8]}...)")
                
            except Exception as e:
                print(f"❌ Error: {file_path.name} - {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"[{layer_name.upper()} LAYER] Import complete")
        print(f"Imported {len(self.layer_gids[layer_name])} subgraph(s)")
        print(f"{'='*80}")
    
    def create_trinity_links(self):
        """Create REFERENCE relationships between the three layers"""
        print("\n" + "="*80)
        print("[Trinity Links] Creating cross-layer relationships")
        print("="*80)
        
        total_links = 0
        
        # Bottom -> Middle
        print("\n[Linking] Bottom → Middle")
        for bottom_gid in self.layer_gids['bottom']:
            for middle_gid in self.layer_gids['middle']:
                try:
                    result = ref_link(self.n4j, bottom_gid, middle_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            print(f"  ✅ {bottom_gid[:8]}... → {middle_gid[:8]}...: {count} link(s)")
                except Exception as e:
                    print(f"  ⚠️  Error: {e}")
        
        # Middle -> Top
        print("\n[Linking] Middle → Top")
        for middle_gid in self.layer_gids['middle']:
            for top_gid in self.layer_gids['top']:
                try:
                    result = ref_link(self.n4j, middle_gid, top_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            print(f"  ✅ {middle_gid[:8]}... → {top_gid[:8]}...: {count} link(s)")
                except Exception as e:
                    print(f"  ⚠️  Error: {e}")
        
        print(f"\n{'='*80}")
        print(f"[Trinity Links] Complete")
        print(f"Created {total_links} REFERENCE relationship(s)")
        print(f"{'='*80}")
    
    def print_statistics(self):
        """Print statistics"""
        print("\n" + "="*80)
        print("[Statistics]")
        print("="*80)
        
        # Node count
        result = self.n4j.query("MATCH (n) WHERE NOT n:Summary RETURN count(n) as count")
        node_count = result[0]['count'] if result else 0
        
        # Summary count
        result = self.n4j.query("MATCH (s:Summary) RETURN count(s) as count")
        summary_count = result[0]['count'] if result else 0
        
        # Relationship count
        result = self.n4j.query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result[0]['count'] if result else 0
        
        # REFERENCE count
        result = self.n4j.query("MATCH ()-[r:REFERENCE]->() RETURN count(r) as count")
        ref_count = result[0]['count'] if result else 0
        
        # Entity type breakdown
        result = self.n4j.query("""
            MATCH (n)
            WHERE NOT n:Summary
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print(f"\nOverall:")
        print(f"  - Entity nodes: {node_count}")
        print(f"  - Summary nodes: {summary_count}")
        print(f"  - Total relationships: {rel_count}")
        print(f"  - REFERENCE relationships: {ref_count}")
        
        print(f"\nLayer breakdown:")
        print(f"  - Bottom layer: {len(self.layer_gids['bottom'])} subgraph(s)")
        print(f"  - Middle layer: {len(self.layer_gids['middle'])} subgraph(s)")
        print(f"  - Top layer: {len(self.layer_gids['top'])} subgraph(s)")
        
        print(f"\nEntity types (top 10):")
        for item in result:
            print(f"  - {item['type']}: {item['count']}")
        
        print(f"\n{'='*80}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Three-layer medical knowledge graph import')
    
    # Data paths
    parser.add_argument('--bottom', type=str, help='Bottom layer data path (medical dictionaries)')
    parser.add_argument('--middle', type=str, help='Middle layer data path (clinical guidelines)')
    parser.add_argument('--top', type=str, help='Top layer data path (patient reports)')
    
    # Feature flags
    parser.add_argument('--clear', action='store_true', help='Clear the database before import')
    parser.add_argument('--trinity', action='store_true', help='Create Trinity cross-layer REFERENCE links')
    parser.add_argument('--grained_chunk', action='store_true', help='Use fine-grained chunking')
    parser.add_argument('--ingraphmerge', action='store_true', help='Merge similar nodes within the graph')
    
    # Neo4j config
    parser.add_argument('--neo4j-url', type=str, 
                       default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str, 
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, 
                       default=os.getenv('NEO4J_PASSWORD'))
    
    args = parser.parse_args()
    
    # Check Neo4j password
    if not args.neo4j_password:
        print("❌ Error: Neo4j password not provided")
        print("Set the NEO4J_PASSWORD environment variable or use --neo4j-password")
        sys.exit(1)
    
    # Initialize importer
    importer = ThreeLayerImporter(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    # Clear database
    if args.clear:
        importer.clear_database()
    
    # Import each layer
    if args.bottom:
        importer.import_layer('bottom', args.bottom, args)
    
    if args.middle:
        importer.import_layer('middle', args.middle, args)
    
    if args.top:
        importer.import_layer('top', args.top, args)
    
    # Create Trinity links
    if args.trinity:
        importer.create_trinity_links()
    
    # Print statistics
    importer.print_statistics()
    
    print("\n🎉 All tasks complete!")


if __name__ == '__main__':
    main()
