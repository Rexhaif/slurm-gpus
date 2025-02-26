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
    """Run scontrol command and return its JSON output"""
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except FileNotFoundError:
        console = Console()
        console.print("[bold red]Error:[/bold red] scontrol command not found. Please ensure SLURM is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        console = Console()
        console.print(f"[bold red]Error running scontrol:[/bold red] {e}")
        console.print(f"stderr: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console = Console()
        console.print(f"[bold red]Error parsing JSON output:[/bold red] {e}")
        sys.exit(1)

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
        "nvidia_tesla_t4": 16
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
    if "gres_used" in node_data and node_data["gres_used"]:
        used_match = re.search(r'gpu:[^:]+:\d+\(IDX:([^)]+)\)', node_data["gres_used"])
        if used_match:
            idx_str = used_match.group(1)
            # Parse individual indexes or ranges like 0-3
            for part in idx_str.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    gpu_info["allocated_indexes"].extend(list(range(start, end + 1)))
                else:
                    gpu_info["allocated_indexes"].append(int(part))
    
    gpu_info["allocated_gpus"] = len(gpu_info["allocated_indexes"])
    
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
    data = run_scontrol()
    
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
