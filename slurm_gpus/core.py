#!/usr/bin/env python3

import json
import subprocess
import re
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

def run_scontrol():
    """Run scontrol command and return its output, trying JSON first, then plain text"""
    console = Console()
    
    # First try with JSON format (newer SLURM versions)
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return {"format": "json", "data": data}
    except subprocess.CalledProcessError as e:
        if "invalid option" in e.stderr.lower() or "unrecognized option" in e.stderr.lower():
            # JSON format not supported, try plain text
            console.print("[yellow]JSON format not supported, falling back to plain text parsing...[/yellow]")
        else:
            console.print(f"[bold red]Error running scontrol:[/bold red] {e}")
            console.print(f"stderr: {e.stderr}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error parsing JSON output:[/bold red] {e}")
        console.print("[yellow]Falling back to plain text parsing...[/yellow]")
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] scontrol command not found. Please ensure SLURM is installed and in your PATH.")
        sys.exit(1)
    
    # Fall back to plain text format (older SLURM versions)
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"format": "text", "data": result.stdout}
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running scontrol in plain text mode:[/bold red] {e}")
        console.print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] scontrol command not found. Please ensure SLURM is installed and in your PATH.")
        sys.exit(1)

def parse_plain_text_output(text_output):
    """Parse plain text output from scontrol and convert to structured format"""
    nodes = []
    
    # Split by NodeName= to get individual node blocks
    # Add 'NodeName=' back to each block except the first empty one
    node_blocks = text_output.split("NodeName=")
    node_blocks = [block for block in node_blocks if block.strip()]
    
    for block in node_blocks:
        node_data = {}
        
        # Extract node name (first part of the block before space)
        node_name_match = re.match(r'([^\s]+)', block)
        if node_name_match:
            node_data["name"] = node_name_match.group(1)
        
        # Extract state
        state_match = re.search(r'State=(\S+)', block)
        if state_match:
            # Store state as a list to match JSON format
            node_data["state"] = [state_match.group(1)]
        
        # Extract partitions
        partitions_match = re.search(r'Partitions=(\S+)', block)
        if partitions_match:
            node_data["partitions"] = partitions_match.group(1).split(",")
        
        # Extract GPU information from Gres
        gres_match = re.search(r'Gres=([^\n]+)', block)
        if gres_match and "gpu:" in gres_match.group(1):
            node_data["gres"] = gres_match.group(1)
        
        # Extract allocated GPU information
        tres_match = re.search(r'AllocTRES=([^\n]+)', block)
        if tres_match and "gres/gpu" in tres_match.group(1):
            node_data["gres_used"] = tres_match.group(1)
            
            # Try to extract GPU indexes from the block
            gpu_idx_match = re.search(r'gpu:[^:]+:\d+\(IDX:([^)]+)\)', block)
            if gpu_idx_match:
                # Create a fake gres_used with IDX format for compatibility
                gpu_model_match = re.search(r'gpu:([^:]+):', node_data["gres"])
                gpu_count_match = re.search(r'gpu:[^:]+:(\d+)', node_data["gres"])
                
                if gpu_model_match and gpu_count_match:
                    model = gpu_model_match.group(1)
                    count = gpu_count_match.group(1)
                    node_data["gres_used"] = f"gpu:{model}:{count}(IDX:{gpu_idx_match.group(1)})"
        
        nodes.append(node_data)
    
    return {"nodes": nodes}

def get_gpu_memory_size(gpu_model, partitions=None):
    """Get GPU memory size using internal dictionary or partition names.
    
    First tries to extract memory from partition names like 'gpu-vram-12gb',
    then falls back to internal dictionary of known GPU models.
    """
    # Dictionary of known GPU models and their VRAM sizes in GB
    gpu_memory_dict = {
        # NVIDIA Gaming/Professional GPUs
        "nvidia_geforce_rtx_2080_ti": 11,
        "nvidia_geforce_rtx_3080": 10,
        "nvidia_geforce_rtx_3090": 24,
        "nvidia_geforce_rtx_4090": 24,
        
        # NVIDIA Quadro/RTX Professional GPUs
        "nvidia_quadro_rtx_4000": 8,
        "nvidia_quadro_rtx_5000": 16,
        "nvidia_quadro_rtx_6000": 24,
        "nvidia_quadro_rtx_8000": 48,
        "nvidia_rtx_a2000": 6,
        "nvidia_rtx_a4000": 16,
        "nvidia_rtx_a5000": 24,
        "nvidia_rtx_a6000": 48,
        "nvidia_rtx_a8000": 48,
        "nvidia_rtx_5000_ada_generation": 32,
        "nvidia_rtx_6000_ada_generation": 48,
        
        # NVIDIA Data Center GPUs
        "nvidia_a10": 24,
        "nvidia_a30": 24,
        "nvidia_a40": 48,
        "nvidia_a100": 40,
        "nvidia_a100_80gb": 80,
        "nvidia_h100": 80,
        "nvidia_h100_nvl": 94,  # NVL variant as per partitions
        "nvidia_l4": 24,
        "nvidia_l40": 48,
        "nvidia_l40s": 48,
        
        # Older NVIDIA GPUs
        "nvidia_tesla_k80": 12,
        "nvidia_tesla_p4": 8,
        "nvidia_tesla_p40": 24,
        "nvidia_tesla_p100": 16,
        "nvidia_tesla_v100": 32,
        "nvidia_tesla_v100_32gb": 32,
        "nvidia_tesla_t4": 16,
        
        # Add common GPU models without nvidia_ prefix for plain text output
        "a100": 40,
        "a100_80gb": 80,
        "v100": 32,
        "v100_32gb": 32,
        "p100": 16,
        "t4": 16,
        "a10": 24,
        "a30": 24,
        "a40": 48,
        "h100": 80,
        "l4": 24,
        "l40": 48,
        "l40s": 48
    }
    
    # First try to get memory from partition names
    if partitions:
        for partition in partitions:
            match = re.search(r'gpu-vram-(\d+)gb', partition)
            if match:
                return int(match.group(1))
    
    # Fall back to the dictionary if memory not found in partition names
    normalized_model = gpu_model.lower() if gpu_model else ""
    return gpu_memory_dict.get(normalized_model)

def parse_gpu_info(node_data):
    """Parse GPU information from a node's data"""
    # Skip nodes without GPUs
    if "gres" not in node_data or not node_data["gres"] or "gpu:" not in node_data["gres"]:
        return None
    
    # Extract GPU info
    gpu_info = {
        "node_name": node_data["name"],
        "gpu_model": None,
        "total_gpus": 0,
        "allocated_gpus": 0,
        "available_gpus": 0,
        "memory_per_gpu": None,
        "allocated_indexes": [],
        "available_indexes": [],
        "state": node_data.get("state", ["UNKNOWN"])[0]
    }
    
    # Extract GPU model and count
    gres_match = re.search(r'gpu:([^:]+):(\d+)', node_data["gres"])
    if gres_match:
        gpu_info["gpu_model"] = gres_match.group(1)
        gpu_info["total_gpus"] = int(gres_match.group(2))
    
    # Get memory size from partition names or fallback to built-in dictionary
    partitions = node_data.get("partitions", [])
    gpu_info["memory_per_gpu"] = get_gpu_memory_size(gpu_info["gpu_model"], partitions)
    
    # Extract allocated GPUs
    allocated_gpus = 0
    if "gres_used" in node_data and node_data["gres_used"]:
        # Try to extract directly from gres_used
        used_match = re.search(r'gpu:[^:]+:(\d+)', node_data["gres_used"])
        if used_match:
            allocated_gpus = int(used_match.group(1))
        
        # Try to extract GPU indexes
        idx_match = re.search(r'IDX:([^)]+)', node_data["gres_used"])
        if idx_match:
            idx_str = idx_match.group(1)
            # Skip if it's N/A
            if idx_str.strip() != "N/A":
                # Parse individual indexes or ranges like 0-3
                for part in idx_str.split(','):
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            gpu_info["allocated_indexes"].extend(list(range(start, end + 1)))
                        except ValueError:
                            pass  # Skip if not valid integers
                    else:
                        try:
                            gpu_info["allocated_indexes"].append(int(part))
                        except ValueError:
                            pass  # Skip if not a valid integer
        
        # If we have specific indexes, count them; otherwise use the extracted count
        if gpu_info["allocated_indexes"]:
            allocated_gpus = len(gpu_info["allocated_indexes"])
        
        # Alternative extraction for older formats
        if not gpu_info["allocated_indexes"] and "gres/gpu=" in node_data["gres_used"]:
            alloc_match = re.search(r'gres/gpu=(\d+)', node_data["gres_used"])
            if alloc_match:
                allocated_gpus = int(alloc_match.group(1))
                
                # If we have allocated GPUs but no indexes, assume they're the first n GPUs
                if allocated_gpus > 0:
                    gpu_info["allocated_indexes"] = list(range(min(allocated_gpus, gpu_info["total_gpus"])))
    
    gpu_info["allocated_gpus"] = allocated_gpus if allocated_gpus > 0 else len(gpu_info["allocated_indexes"])
    
    # Calculate available GPUs and their indexes
    all_indexes = set(range(gpu_info["total_gpus"]))
    allocated_set = set(gpu_info["allocated_indexes"])
    gpu_info["available_indexes"] = sorted(list(all_indexes - allocated_set))
    gpu_info["available_gpus"] = len(gpu_info["available_indexes"])
    
    return gpu_info

def format_gpu_indexes(indexes, color):
    """Format GPU indexes with better representation of ranges"""
    if not indexes:
        return ""
    
    # Convert list of indexes to ranges when possible
    ranges = []
    start = end = indexes[0]
    
    for i in range(1, len(indexes)):
        if indexes[i] == end + 1:
            end = indexes[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = indexes[i]
    
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return f"[{color}]{', '.join(ranges)}[/{color}]"

def display_gpu_info(gpu_data):
    """Display GPU information using Rich library"""
    console = Console()
    
    # Create summary table
    summary_table = Table(title="GPU Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Type", style="cyan")
    summary_table.add_column("Memory", style="green")
    summary_table.add_column("Total", style="blue")
    summary_table.add_column("Allocated", style="yellow")
    summary_table.add_column("Available", style="green")
    summary_table.add_column("% Free", style="blue")
    
    # Group by GPU model and memory
    gpu_summary = {}
    for gpu in gpu_data:
        if not gpu:
            continue
            
        key = (gpu["gpu_model"], gpu["memory_per_gpu"])
        if key not in gpu_summary:
            gpu_summary[key] = {
                "total": 0,
                "allocated": 0,
                "available": 0
            }
        
        gpu_summary[key]["total"] += gpu["total_gpus"]
        gpu_summary[key]["allocated"] += gpu["allocated_gpus"]
        gpu_summary[key]["available"] += gpu["available_gpus"]
    
    # Add rows to summary table
    for (model, memory), counts in sorted(gpu_summary.items()):
        model_display = model.replace("_", " ").title() if model else "Unknown"
        memory_str = f"{memory} GB" if memory else "Unknown"
        
        # Calculate percentage free
        if counts["total"] > 0:
            percent_free = (counts["available"] / counts["total"]) * 100
            percent_str = f"{percent_free:.1f}%"
            # Color-code the percentage
            if percent_free > 75:
                percent_display = f"[green]{percent_str}[/green]"
            elif percent_free > 25:
                percent_display = f"[yellow]{percent_str}[/yellow]"
            else:
                percent_display = f"[red]{percent_str}[/red]"
        else:
            percent_display = "N/A"
            
        summary_table.add_row(
            model_display,
            memory_str,
            str(counts["total"]),
            f"[yellow]{counts['allocated']}[/yellow]",
            f"[green]{counts['available']}[/green]",
            percent_display
        )
    
    # Create detailed table
    detail_table = Table(title="GPU Allocation Details", show_header=True, header_style="bold magenta")
    detail_table.add_column("Node", style="cyan")
    detail_table.add_column("State", style="blue")
    detail_table.add_column("GPU Type", style="blue")
    detail_table.add_column("Memory", style="green")
    detail_table.add_column("Total", style="blue")
    detail_table.add_column("Allocated", style="yellow")
    detail_table.add_column("Available", style="green")
    detail_table.add_column("Allocated IDs", width=15)
    detail_table.add_column("Available IDs", width=15)
    
    # Add rows to detail table
    for gpu in sorted(gpu_data, key=lambda x: x["node_name"] if x else ""):
        if not gpu:
            continue
            
        memory_str = f"{gpu['memory_per_gpu']} GB" if gpu["memory_per_gpu"] else "Unknown"
        
        # Create colored state text
        state_text = gpu["state"]
        state_style = "green" if state_text == "IDLE" else "yellow" if state_text == "MIXED" else "red"
        
        detail_table.add_row(
            gpu["node_name"],
            f"[{state_style}]{state_text}[/{state_style}]",
            gpu["gpu_model"].replace("_", " ").title(),
            memory_str,
            str(gpu["total_gpus"]),
            f"[yellow]{gpu['allocated_gpus']}[/yellow]",
            f"[green]{gpu['available_gpus']}[/green]",
            format_gpu_indexes(gpu["allocated_indexes"], "yellow"),
            format_gpu_indexes(gpu["available_indexes"], "green")
        )
    
    # Display tables
    console.print(Panel(summary_table, title="GPU Summary", border_style="green"))
    console.print("\n")
    console.print(Panel(detail_table, title="GPU Details", border_style="blue"))

def main():
    """Main function"""
    console = Console()
    
    # Show script header
    console.print(Panel.fit(
        Text("GPU Status Viewer", style="bold cyan", justify="center"),
        subtitle="Displays available and allocated GPUs on SLURM cluster",
        border_style="green"
    ))
    
    console.print("[yellow]Fetching node data from SLURM...[/yellow]")
    
    # Get node data from scontrol
    result = run_scontrol()
    
    # Process data based on format
    if result["format"] == "json":
        data = result["data"]
    else:
        # Parse plain text output
        data = parse_plain_text_output(result["data"])
    
    # Parse GPU information from each node
    gpu_data = []
    for node in data.get("nodes", []):
        gpu_info = parse_gpu_info(node)
        if gpu_info:
            gpu_data.append(gpu_info)
    
    if not gpu_data:
        console.print("[bold yellow]No GPU nodes found in the cluster.[/bold yellow]")
        sys.exit(0)
    
    # Count total GPUs
    total_gpus = sum(gpu["total_gpus"] for gpu in gpu_data if gpu)
    available_gpus = sum(gpu["available_gpus"] for gpu in gpu_data if gpu)
    
    console.print(f"[green]Found [bold]{total_gpus}[/bold] GPUs across [bold]{len(gpu_data)}[/bold] nodes, [bold]{available_gpus}[/bold] available.[/green]")
    
    # Display GPU information
    display_gpu_info(gpu_data)

if __name__ == "__main__":
    main()
