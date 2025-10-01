#!/usr/bin/env python3
"""
Cleanup script for Video Summarizer project.
Removes temporary files, cleans up output directories, and manages logs.
"""

import os
import shutil
import argparse
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSummarizerCleanup:
    """Cleanup utility for Video Summarizer project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.temp_dir = self.project_root / "temp"
        self.output_dir = self.project_root / "output"
        self.logs_dir = self.project_root / "logs"
        self.mlflow_db = self.project_root / "mlflow.db"
        self.model_registry = self.project_root / "model_registry"
        
    def cleanup_temp_files(self, confirm: bool = False) -> dict:
        """Clean up temporary files."""
        results = {"files_removed": 0, "space_freed": 0, "errors": []}
        
        if not self.temp_dir.exists():
            logger.info("Temp directory does not exist")
            return results
            
        temp_files = list(self.temp_dir.glob("*"))
        if not temp_files:
            logger.info("No temporary files to clean up")
            return results
            
        # Calculate total size
        total_size = sum(f.stat().st_size for f in temp_files if f.is_file())
        logger.info(f"Found {len(temp_files)} temporary files ({total_size / (1024*1024):.1f} MB)")
        
        if not confirm:
            logger.info("Use --confirm flag to actually remove files")
            return results
            
        for file_path in temp_files:
            try:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    results["files_removed"] += 1
                    results["space_freed"] += file_size
                    logger.info(f"Removed: {file_path.name}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    logger.info(f"Removed directory: {file_path.name}")
            except Exception as e:
                error_msg = f"Error removing {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                
        logger.info(f"Cleaned up {results['files_removed']} files, freed {results['space_freed'] / (1024*1024):.1f} MB")
        return results
    
    def cleanup_output_files(self, keep_recent_days: int = 7, confirm: bool = False) -> dict:
        """Clean up output files older than specified days."""
        results = {"files_removed": 0, "space_freed": 0, "errors": []}
        
        if not self.output_dir.exists():
            logger.info("Output directory does not exist")
            return results
            
        cutoff_date = datetime.now() - timedelta(days=keep_recent_days)
        
        # Clean up subdirectories
        for subdir in ["audio", "summaries", "transcripts"]:
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                files_to_remove = []
                for file_path in subdir_path.iterdir():
                    if file_path.is_file():
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_date:
                            files_to_remove.append(file_path)
                
                if files_to_remove:
                    total_size = sum(f.stat().st_size for f in files_to_remove)
                    logger.info(f"Found {len(files_to_remove)} old files in {subdir} ({total_size / (1024*1024):.1f} MB)")
                    
                    if confirm:
                        for file_path in files_to_remove:
                            try:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                results["files_removed"] += 1
                                results["space_freed"] += file_size
                                logger.info(f"Removed: {file_path.name}")
                            except Exception as e:
                                error_msg = f"Error removing {file_path}: {str(e)}"
                                results["errors"].append(error_msg)
                                logger.error(error_msg)
                    else:
                        logger.info(f"Use --confirm flag to remove {len(files_to_remove)} old files")
        
        if results["files_removed"] > 0:
            logger.info(f"Cleaned up {results['files_removed']} output files, freed {results['space_freed'] / (1024*1024):.1f} MB")
        
        return results
    
    def cleanup_logs(self, max_size_mb: int = 10, keep_backups: int = 3, confirm: bool = False) -> dict:
        """Clean up and rotate log files."""
        results = {"files_rotated": 0, "files_removed": 0, "space_freed": 0, "errors": []}
        
        if not self.logs_dir.exists():
            logger.info("Logs directory does not exist")
            return results
            
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for log_file in self.logs_dir.glob("*.log"):
            try:
                file_size = log_file.stat().st_size
                
                if file_size > max_size_bytes:
                    logger.info(f"Log file {log_file.name} is {file_size / (1024*1024):.1f} MB (limit: {max_size_mb} MB)")
                    
                    if confirm:
                        # Rotate log file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{log_file.stem}_{timestamp}.log"
                        backup_path = log_file.parent / backup_name
                        
                        # Move current log to backup
                        shutil.move(str(log_file), str(backup_path))
                        results["files_rotated"] += 1
                        results["space_freed"] += file_size
                        
                        # Remove old backups if we have too many
                        backups = sorted(self.logs_dir.glob(f"{log_file.stem}_*.log"), 
                                       key=lambda x: x.stat().st_mtime, reverse=True)
                        if len(backups) > keep_backups:
                            for old_backup in backups[keep_backups:]:
                                old_backup.unlink()
                                results["files_removed"] += 1
                                logger.info(f"Removed old backup: {old_backup.name}")
                        
                        logger.info(f"Rotated log file: {log_file.name} -> {backup_name}")
                    else:
                        logger.info(f"Use --confirm flag to rotate log file {log_file.name}")
                else:
                    logger.info(f"Log file {log_file.name} is within size limits ({file_size / (1024*1024):.1f} MB)")
                    
            except Exception as e:
                error_msg = f"Error processing log file {log_file}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def cleanup_mlflow_data(self, confirm: bool = False) -> dict:
        """Clean up MLflow database and artifacts."""
        results = {"files_removed": 0, "space_freed": 0, "errors": []}
        
        # Check MLflow database
        if self.mlflow_db.exists():
            file_size = self.mlflow_db.stat().st_size
            logger.info(f"MLflow database size: {file_size / (1024*1024):.1f} MB")
            
            if confirm:
                try:
                    self.mlflow_db.unlink()
                    results["files_removed"] += 1
                    results["space_freed"] += file_size
                    logger.info("Removed MLflow database")
                except Exception as e:
                    error_msg = f"Error removing MLflow database: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)
            else:
                logger.info("Use --confirm flag to remove MLflow database")
        
        # Check model registry
        if self.model_registry.exists():
            registry_files = list(self.model_registry.rglob("*"))
            total_size = sum(f.stat().st_size for f in registry_files if f.is_file())
            logger.info(f"Model registry contains {len(registry_files)} files ({total_size / (1024*1024):.1f} MB)")
            
            if confirm and registry_files:
                try:
                    shutil.rmtree(self.model_registry)
                    results["files_removed"] += len(registry_files)
                    results["space_freed"] += total_size
                    logger.info("Removed model registry")
                except Exception as e:
                    error_msg = f"Error removing model registry: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)
            elif registry_files:
                logger.info("Use --confirm flag to remove model registry")
        
        return results
    
    def full_cleanup(self, confirm: bool = False, keep_recent_days: int = 7, 
                    max_log_size_mb: int = 10) -> dict:
        """Perform full cleanup of the project."""
        logger.info("Starting full cleanup...")
        
        total_results = {
            "temp_files": self.cleanup_temp_files(confirm),
            "output_files": self.cleanup_output_files(keep_recent_days, confirm),
            "logs": self.cleanup_logs(max_log_size_mb, confirm=confirm),
            "mlflow_data": self.cleanup_mlflow_data(confirm)
        }
        
        # Calculate totals
        total_files = sum(result["files_removed"] for result in total_results.values())
        total_space = sum(result["space_freed"] for result in total_results.values())
        total_errors = sum(len(result["errors"]) for result in total_results.values())
        
        logger.info(f"Cleanup complete! Removed {total_files} files, freed {total_space / (1024*1024):.1f} MB")
        if total_errors > 0:
            logger.warning(f"Encountered {total_errors} errors during cleanup")
        
        return total_results

def main():
    parser = argparse.ArgumentParser(description="Clean up Video Summarizer project files")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--confirm", action="store_true", help="Actually perform cleanup (dry run by default)")
    parser.add_argument("--temp", action="store_true", help="Clean up temporary files only")
    parser.add_argument("--output", action="store_true", help="Clean up output files only")
    parser.add_argument("--logs", action="store_true", help="Clean up log files only")
    parser.add_argument("--mlflow", action="store_true", help="Clean up MLflow data only")
    parser.add_argument("--keep-recent-days", type=int, default=7, help="Keep output files from last N days")
    parser.add_argument("--max-log-size-mb", type=int, default=10, help="Maximum log file size in MB")
    
    args = parser.parse_args()
    
    cleanup = VideoSummarizerCleanup(args.project_root)
    
    if args.temp:
        cleanup.cleanup_temp_files(args.confirm)
    elif args.output:
        cleanup.cleanup_output_files(args.keep_recent_days, args.confirm)
    elif args.logs:
        cleanup.cleanup_logs(args.max_log_size_mb, confirm=args.confirm)
    elif args.mlflow:
        cleanup.cleanup_mlflow_data(args.confirm)
    else:
        # Full cleanup
        cleanup.full_cleanup(args.confirm, args.keep_recent_days, args.max_log_size_mb)

if __name__ == "__main__":
    main()
