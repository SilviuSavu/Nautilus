"""
Strategy Version Management System
Git-like versioning for trading strategies with branch management, merge conflict resolution, and release tagging
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Version status"""
    DRAFT = "draft"
    COMMITTED = "committed"
    TAGGED = "tagged"
    ARCHIVED = "archived"


class BranchType(Enum):
    """Branch types"""
    MAIN = "main"
    DEVELOP = "develop"
    FEATURE = "feature"
    RELEASE = "release"
    HOTFIX = "hotfix"


class MergeStrategy(Enum):
    """Merge strategies"""
    FAST_FORWARD = "fast_forward"
    THREE_WAY = "three_way"
    SQUASH = "squash"


class ConflictType(Enum):
    """Types of merge conflicts"""
    CODE_CONFLICT = "code_conflict"
    PARAMETER_CONFLICT = "parameter_conflict"
    CONFIG_CONFLICT = "config_conflict"


class VersionInfo(BaseModel):
    """Version information"""
    version_id: str
    strategy_id: str
    version_number: str
    branch_name: str
    parent_version: Optional[str] = None
    merge_base: Optional[str] = None
    
    # Content
    strategy_code: str
    strategy_config: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    
    # Version details
    commit_hash: str
    commit_message: str
    author: str
    created_at: datetime
    status: VersionStatus = VersionStatus.DRAFT
    
    # Tags and labels
    tags: List[str] = []
    labels: Dict[str, str] = {}


class BranchInfo(BaseModel):
    """Branch information"""
    branch_name: str
    branch_type: BranchType
    base_branch: str
    head_version: str
    created_by: str
    created_at: datetime
    last_commit_at: Optional[datetime] = None
    is_active: bool = True
    merge_target: Optional[str] = None


class MergeConflict(BaseModel):
    """Merge conflict information"""
    conflict_id: str
    conflict_type: ConflictType
    source_branch: str
    target_branch: str
    source_version: str
    target_version: str
    
    # Conflict details
    conflict_path: str  # e.g., "parameters.fast_period", "code.line_45"
    source_value: Any
    target_value: Any
    base_value: Optional[Any] = None
    
    # Resolution
    resolved: bool = False
    resolution_value: Optional[Any] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class MergeRequest(BaseModel):
    """Merge request"""
    merge_id: str
    source_branch: str
    target_branch: str
    source_version: str
    target_version: str
    merge_strategy: MergeStrategy
    
    # Details
    title: str
    description: str
    created_by: str
    created_at: datetime
    
    # Status
    conflicts: List[MergeConflict] = []
    has_conflicts: bool = False
    merged: bool = False
    merged_version: Optional[str] = None
    merged_at: Optional[datetime] = None


class ReleaseTag(BaseModel):
    """Release tag information"""
    tag_name: str
    version_id: str
    release_type: str  # 'major', 'minor', 'patch', 'hotfix'
    release_notes: str
    created_by: str
    created_at: datetime
    pre_release: bool = False


class VersionControl:
    """Git-like version control system for trading strategies"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path("/tmp/strategy_versions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory state (in production, this would be database-backed)
        self._versions: Dict[str, VersionInfo] = {}
        self._branches: Dict[str, BranchInfo] = {}
        self._merge_requests: Dict[str, MergeRequest] = {}
        self._release_tags: Dict[str, ReleaseTag] = {}
        
        # Initialize default branches
        self._initialize_default_branches()
    
    def _initialize_default_branches(self):
        """Initialize default branches"""
        main_branch = BranchInfo(
            branch_name="main",
            branch_type=BranchType.MAIN,
            base_branch="",
            head_version="",
            created_by="system",
            created_at=datetime.utcnow()
        )
        
        develop_branch = BranchInfo(
            branch_name="develop",
            branch_type=BranchType.DEVELOP,
            base_branch="main",
            head_version="",
            created_by="system",
            created_at=datetime.utcnow()
        )
        
        self._branches["main"] = main_branch
        self._branches["develop"] = develop_branch
    
    def commit_version(self, 
                       strategy_id: str,
                       strategy_code: str,
                       strategy_config: Dict[str, Any],
                       branch_name: str = "main",
                       commit_message: str = "Update strategy",
                       author: str = "system",
                       metadata: Dict[str, Any] = None) -> VersionInfo:
        """Commit a new version of a strategy"""
        
        # Generate version ID and hash
        version_id = str(uuid.uuid4())
        content_hash = self._calculate_hash(strategy_code, strategy_config)
        
        # Get parent version
        branch = self._branches.get(branch_name)
        parent_version = branch.head_version if branch and branch.head_version else None
        
        # Generate version number
        version_number = self._generate_version_number(strategy_id, branch_name, parent_version)
        
        version_info = VersionInfo(
            version_id=version_id,
            strategy_id=strategy_id,
            version_number=version_number,
            branch_name=branch_name,
            parent_version=parent_version,
            strategy_code=strategy_code,
            strategy_config=strategy_config,
            metadata=metadata or {},
            commit_hash=content_hash,
            commit_message=commit_message,
            author=author,
            created_at=datetime.utcnow(),
            status=VersionStatus.COMMITTED
        )
        
        # Store version
        self._versions[version_id] = version_info
        
        # Update branch head
        if branch:
            branch.head_version = version_id
            branch.last_commit_at = datetime.utcnow()
        
        # Save to persistent storage
        self._save_version_to_storage(version_info)
        
        logger.info(f"Committed version {version_number} for strategy {strategy_id} on branch {branch_name}")
        return version_info
    
    def create_branch(self, 
                      branch_name: str,
                      base_branch: str = "main",
                      branch_type: BranchType = BranchType.FEATURE,
                      created_by: str = "system") -> BranchInfo:
        """Create a new branch"""
        
        if branch_name in self._branches:
            raise ValueError(f"Branch {branch_name} already exists")
        
        base_branch_info = self._branches.get(base_branch)
        if not base_branch_info:
            raise ValueError(f"Base branch {base_branch} not found")
        
        branch_info = BranchInfo(
            branch_name=branch_name,
            branch_type=branch_type,
            base_branch=base_branch,
            head_version=base_branch_info.head_version,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        self._branches[branch_name] = branch_info
        
        logger.info(f"Created branch {branch_name} from {base_branch}")
        return branch_info
    
    def create_merge_request(self, 
                             source_branch: str,
                             target_branch: str,
                             title: str,
                             description: str = "",
                             merge_strategy: MergeStrategy = MergeStrategy.THREE_WAY,
                             created_by: str = "system") -> MergeRequest:
        """Create a merge request"""
        
        source_branch_info = self._branches.get(source_branch)
        target_branch_info = self._branches.get(target_branch)
        
        if not source_branch_info or not target_branch_info:
            raise ValueError("Source or target branch not found")
        
        if not source_branch_info.head_version or not target_branch_info.head_version:
            raise ValueError("Source or target branch has no commits")
        
        merge_id = str(uuid.uuid4())
        
        merge_request = MergeRequest(
            merge_id=merge_id,
            source_branch=source_branch,
            target_branch=target_branch,
            source_version=source_branch_info.head_version,
            target_version=target_branch_info.head_version,
            merge_strategy=merge_strategy,
            title=title,
            description=description,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        # Check for conflicts
        conflicts = self._detect_merge_conflicts(
            source_branch_info.head_version,
            target_branch_info.head_version
        )
        
        merge_request.conflicts = conflicts
        merge_request.has_conflicts = len(conflicts) > 0
        
        self._merge_requests[merge_id] = merge_request
        
        logger.info(f"Created merge request {merge_id} from {source_branch} to {target_branch}")
        return merge_request
    
    def _detect_merge_conflicts(self, 
                                source_version_id: str, 
                                target_version_id: str) -> List[MergeConflict]:
        """Detect merge conflicts between two versions"""
        
        source_version = self._versions.get(source_version_id)
        target_version = self._versions.get(target_version_id)
        
        if not source_version or not target_version:
            return []
        
        conflicts = []
        
        # Find common ancestor (merge base)
        merge_base = self._find_merge_base(source_version_id, target_version_id)
        base_version = self._versions.get(merge_base) if merge_base else None
        
        # Check for code conflicts
        if source_version.strategy_code != target_version.strategy_code:
            # Simple conflict detection - in practice, this would be more sophisticated
            conflict = MergeConflict(
                conflict_id=str(uuid.uuid4()),
                conflict_type=ConflictType.CODE_CONFLICT,
                source_branch=source_version.branch_name,
                target_branch=target_version.branch_name,
                source_version=source_version_id,
                target_version=target_version_id,
                conflict_path="strategy_code",
                source_value=source_version.strategy_code[:100] + "..." if len(source_version.strategy_code) > 100 else source_version.strategy_code,
                target_value=target_version.strategy_code[:100] + "..." if len(target_version.strategy_code) > 100 else target_version.strategy_code,
                base_value=base_version.strategy_code[:100] + "..." if base_version and len(base_version.strategy_code) > 100 else base_version.strategy_code if base_version else None
            )
            conflicts.append(conflict)
        
        # Check for parameter conflicts
        source_params = source_version.strategy_config.get("parameters", {})
        target_params = target_version.strategy_config.get("parameters", {})
        base_params = base_version.strategy_config.get("parameters", {}) if base_version else {}
        
        all_param_keys = set(source_params.keys()) | set(target_params.keys()) | set(base_params.keys())
        
        for param_key in all_param_keys:
            source_val = source_params.get(param_key)
            target_val = target_params.get(param_key)
            base_val = base_params.get(param_key)
            
            # Conflict if both sides changed the same parameter differently
            if (source_val != base_val and target_val != base_val and source_val != target_val):
                conflict = MergeConflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.PARAMETER_CONFLICT,
                    source_branch=source_version.branch_name,
                    target_branch=target_version.branch_name,
                    source_version=source_version_id,
                    target_version=target_version_id,
                    conflict_path=f"parameters.{param_key}",
                    source_value=source_val,
                    target_value=target_val,
                    base_value=base_val
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def resolve_merge_conflict(self, 
                               merge_id: str,
                               conflict_id: str,
                               resolution_value: Any,
                               resolved_by: str = "system") -> bool:
        """Resolve a merge conflict"""
        
        merge_request = self._merge_requests.get(merge_id)
        if not merge_request:
            return False
        
        conflict = next((c for c in merge_request.conflicts if c.conflict_id == conflict_id), None)
        if not conflict:
            return False
        
        conflict.resolved = True
        conflict.resolution_value = resolution_value
        conflict.resolved_by = resolved_by
        conflict.resolved_at = datetime.utcnow()
        
        # Check if all conflicts are resolved
        merge_request.has_conflicts = any(not c.resolved for c in merge_request.conflicts)
        
        logger.info(f"Resolved conflict {conflict_id} in merge request {merge_id}")
        return True
    
    def execute_merge(self, merge_id: str, merged_by: str = "system") -> Optional[VersionInfo]:
        """Execute a merge request"""
        
        merge_request = self._merge_requests.get(merge_id)
        if not merge_request:
            raise ValueError(f"Merge request {merge_id} not found")
        
        if merge_request.has_conflicts:
            raise ValueError(f"Merge request has unresolved conflicts")
        
        if merge_request.merged:
            return self._versions.get(merge_request.merged_version)
        
        # Get source and target versions
        source_version = self._versions.get(merge_request.source_version)
        target_version = self._versions.get(merge_request.target_version)
        
        if not source_version or not target_version:
            raise ValueError("Source or target version not found")
        
        # Create merged version
        merged_strategy_code = source_version.strategy_code
        merged_strategy_config = source_version.strategy_config.copy()
        
        # Apply conflict resolutions
        for conflict in merge_request.conflicts:
            if conflict.resolved and conflict.resolution_value is not None:
                if conflict.conflict_path.startswith("parameters."):
                    param_key = conflict.conflict_path.split(".", 1)[1]
                    if "parameters" not in merged_strategy_config:
                        merged_strategy_config["parameters"] = {}
                    merged_strategy_config["parameters"][param_key] = conflict.resolution_value
                elif conflict.conflict_path == "strategy_code":
                    merged_strategy_code = conflict.resolution_value
        
        # Commit merged version
        merged_version = self.commit_version(
            strategy_id=source_version.strategy_id,
            strategy_code=merged_strategy_code,
            strategy_config=merged_strategy_config,
            branch_name=merge_request.target_branch,
            commit_message=f"Merge {merge_request.source_branch} into {merge_request.target_branch}",
            author=merged_by,
            metadata={
                "merge_id": merge_id,
                "source_branch": merge_request.source_branch,
                "source_version": merge_request.source_version,
                "merge_strategy": merge_request.merge_strategy.value
            }
        )
        
        # Set merge base
        merged_version.merge_base = self._find_merge_base(
            merge_request.source_version,
            merge_request.target_version
        )
        
        # Update merge request
        merge_request.merged = True
        merge_request.merged_version = merged_version.version_id
        merge_request.merged_at = datetime.utcnow()
        
        logger.info(f"Executed merge {merge_id}, created version {merged_version.version_id}")
        return merged_version
    
    def create_release_tag(self, 
                           version_id: str,
                           tag_name: str,
                           release_type: str = "minor",
                           release_notes: str = "",
                           created_by: str = "system",
                           pre_release: bool = False) -> ReleaseTag:
        """Create a release tag"""
        
        version = self._versions.get(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        if tag_name in self._release_tags:
            raise ValueError(f"Release tag {tag_name} already exists")
        
        release_tag = ReleaseTag(
            tag_name=tag_name,
            version_id=version_id,
            release_type=release_type,
            release_notes=release_notes,
            created_by=created_by,
            created_at=datetime.utcnow(),
            pre_release=pre_release
        )
        
        # Update version status
        version.status = VersionStatus.TAGGED
        version.tags.append(tag_name)
        
        self._release_tags[tag_name] = release_tag
        
        logger.info(f"Created release tag {tag_name} for version {version_id}")
        return release_tag
    
    def _find_merge_base(self, version_id1: str, version_id2: str) -> Optional[str]:
        """Find common ancestor (merge base) of two versions"""
        
        # Get ancestry chains
        ancestors1 = self._get_version_ancestry(version_id1)
        ancestors2 = self._get_version_ancestry(version_id2)
        
        # Find common ancestors
        common_ancestors = set(ancestors1) & set(ancestors2)
        
        if not common_ancestors:
            return None
        
        # Return the most recent common ancestor
        # This is simplified - proper implementation would consider commit timestamps
        return next(iter(common_ancestors))
    
    def _get_version_ancestry(self, version_id: str) -> List[str]:
        """Get ancestry chain of a version"""
        ancestry = []
        current = version_id
        
        while current:
            ancestry.append(current)
            version = self._versions.get(current)
            current = version.parent_version if version else None
        
        return ancestry
    
    def _generate_version_number(self, 
                                 strategy_id: str, 
                                 branch_name: str, 
                                 parent_version: Optional[str]) -> str:
        """Generate version number"""
        
        if not parent_version:
            return "1.0.0"
        
        parent = self._versions.get(parent_version)
        if not parent:
            return "1.0.0"
        
        # Parse parent version number
        try:
            major, minor, patch = map(int, parent.version_number.split("."))
            
            # Increment based on branch type
            if branch_name == "main":
                patch += 1
            elif branch_name.startswith("feature/"):
                minor += 1
                patch = 0
            elif branch_name.startswith("release/"):
                major += 1
                minor = 0
                patch = 0
            else:
                patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except (ValueError, AttributeError):
            # Fallback if parent version number is malformed
            return "1.0.1"
    
    def _calculate_hash(self, strategy_code: str, strategy_config: Dict[str, Any]) -> str:
        """Calculate content hash"""
        content = strategy_code + json.dumps(strategy_config, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _save_version_to_storage(self, version: VersionInfo):
        """Save version to persistent storage"""
        try:
            version_file = self.storage_path / f"{version.version_id}.json"
            with open(version_file, 'w') as f:
                json.dump(version.dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save version {version.version_id}: {str(e)}")
    
    # Query Methods
    def get_version(self, strategy_id: str, version_identifier: str) -> Optional[VersionInfo]:
        """Get version by ID, tag name, or version number"""
        
        # Try by version ID first
        if version_identifier in self._versions:
            return self._versions[version_identifier]
        
        # Try by tag name
        release_tag = self._release_tags.get(version_identifier)
        if release_tag:
            return self._versions.get(release_tag.version_id)
        
        # Try by version number
        for version in self._versions.values():
            if (version.strategy_id == strategy_id and 
                version.version_number == version_identifier):
                return version
        
        return None
    
    def list_versions(self, 
                      strategy_id: str,
                      branch_name: str = None,
                      status: VersionStatus = None) -> List[VersionInfo]:
        """List versions with optional filtering"""
        
        versions = [v for v in self._versions.values() if v.strategy_id == strategy_id]
        
        if branch_name:
            versions = [v for v in versions if v.branch_name == branch_name]
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def list_branches(self, active_only: bool = True) -> List[BranchInfo]:
        """List branches"""
        branches = list(self._branches.values())
        
        if active_only:
            branches = [b for b in branches if b.is_active]
        
        return sorted(branches, key=lambda b: b.created_at, reverse=True)
    
    def get_branch(self, branch_name: str) -> Optional[BranchInfo]:
        """Get branch by name"""
        return self._branches.get(branch_name)
    
    def list_merge_requests(self, 
                            source_branch: str = None,
                            target_branch: str = None,
                            has_conflicts: bool = None) -> List[MergeRequest]:
        """List merge requests with optional filtering"""
        
        merge_requests = list(self._merge_requests.values())
        
        if source_branch:
            merge_requests = [mr for mr in merge_requests if mr.source_branch == source_branch]
        
        if target_branch:
            merge_requests = [mr for mr in merge_requests if mr.target_branch == target_branch]
        
        if has_conflicts is not None:
            merge_requests = [mr for mr in merge_requests if mr.has_conflicts == has_conflicts]
        
        return sorted(merge_requests, key=lambda mr: mr.created_at, reverse=True)
    
    def get_merge_request(self, merge_id: str) -> Optional[MergeRequest]:
        """Get merge request by ID"""
        return self._merge_requests.get(merge_id)
    
    def list_release_tags(self, pre_release: bool = None) -> List[ReleaseTag]:
        """List release tags"""
        tags = list(self._release_tags.values())
        
        if pre_release is not None:
            tags = [t for t in tags if t.pre_release == pre_release]
        
        return sorted(tags, key=lambda t: t.created_at, reverse=True)
    
    def get_release_tag(self, tag_name: str) -> Optional[ReleaseTag]:
        """Get release tag by name"""
        return self._release_tags.get(tag_name)
    
    def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """Delete a branch"""
        
        if branch_name in ["main", "develop"]:
            raise ValueError("Cannot delete main or develop branches")
        
        branch = self._branches.get(branch_name)
        if not branch:
            return False
        
        # Check if branch has unmerged changes
        if not force:
            # Implementation would check for unmerged commits
            pass
        
        branch.is_active = False
        del self._branches[branch_name]
        
        logger.info(f"Deleted branch {branch_name}")
        return True
    
    def get_version_diff(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Get diff between two versions"""
        
        version1 = self._versions.get(version_id1)
        version2 = self._versions.get(version_id2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        diff = {
            "code_changed": version1.strategy_code != version2.strategy_code,
            "config_changed": version1.strategy_config != version2.strategy_config,
            "version1": {
                "id": version1.version_id,
                "number": version1.version_number,
                "branch": version1.branch_name,
                "commit_hash": version1.commit_hash
            },
            "version2": {
                "id": version2.version_id,
                "number": version2.version_number,
                "branch": version2.branch_name,
                "commit_hash": version2.commit_hash
            }
        }
        
        # Parameter differences
        params1 = version1.strategy_config.get("parameters", {})
        params2 = version2.strategy_config.get("parameters", {})
        
        all_param_keys = set(params1.keys()) | set(params2.keys())
        parameter_changes = {}
        
        for key in all_param_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)
            
            if val1 != val2:
                parameter_changes[key] = {
                    "old_value": val1,
                    "new_value": val2,
                    "change_type": "modified" if key in params1 and key in params2 
                                  else "added" if key in params2 
                                  else "removed"
                }
        
        diff["parameter_changes"] = parameter_changes
        
        return diff
    
    def get_version_control_statistics(self) -> Dict[str, Any]:
        """Get version control statistics"""
        
        total_versions = len(self._versions)
        active_branches = len([b for b in self._branches.values() if b.is_active])
        pending_merge_requests = len([mr for mr in self._merge_requests.values() if not mr.merged])
        total_releases = len(self._release_tags)
        
        # Branch statistics
        branch_stats = {}
        for branch_type in BranchType:
            count = len([b for b in self._branches.values() if b.branch_type == branch_type and b.is_active])
            branch_stats[branch_type.value] = count
        
        return {
            "total_versions": total_versions,
            "active_branches": active_branches,
            "pending_merge_requests": pending_merge_requests,
            "total_releases": total_releases,
            "branch_statistics": branch_stats,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
version_control = VersionControl()