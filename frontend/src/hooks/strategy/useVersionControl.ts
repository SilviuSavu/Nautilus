/**
 * Strategy Version Control Hook
 * Manages Git-like version control for strategies
 */

import { useState, useCallback, useEffect } from 'react';
import { message } from 'antd';

export interface VersionInfo {
  versionId: string;
  strategyId: string;
  version: string;
  branch: string;
  commit: string;
  message: string;
  author: string;
  timestamp: Date;
  isActive: boolean;
  tags: string[];
  strategyCode: string;
  strategyConfig: Record<string, any>;
  dependencies: string[];
  metadata: Record<string, any>;
}

export interface Branch {
  branchId: string;
  name: string;
  strategyId: string;
  baseVersion?: string;
  headVersion: string;
  isActive: boolean;
  isProtected: boolean;
  createdBy: string;
  createdAt: Date;
  lastModified: Date;
  description?: string;
}

export interface ChangeSet {
  changeId: string;
  versionId: string;
  changeType: 'added' | 'modified' | 'deleted' | 'renamed';
  filePath: string;
  oldContent?: string;
  newContent?: string;
  lineChanges: {
    added: number;
    removed: number;
    modified: number;
  };
}

export interface MergeRequest {
  mergeId: string;
  sourceBranch: string;
  targetBranch: string;
  strategyId: string;
  title: string;
  description: string;
  requestedBy: string;
  requestedAt: Date;
  status: 'open' | 'merged' | 'closed' | 'conflict';
  reviewers: string[];
  approvals: Array<{
    reviewer: string;
    approvedAt: Date;
    comments?: string;
  }>;
  conflicts: Array<{
    filePath: string;
    conflictType: string;
    description: string;
  }>;
}

export interface ComparisonResult {
  summary: {
    filesChanged: number;
    linesAdded: number;
    linesRemoved: number;
    linesModified: number;
  };
  changes: ChangeSet[];
  conflicts: Array<{
    filePath: string;
    type: string;
    description: string;
  }>;
}

export interface TagInfo {
  tagId: string;
  name: string;
  versionId: string;
  description?: string;
  taggedBy: string;
  taggedAt: Date;
  isRelease: boolean;
  metadata: Record<string, any>;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

export const useVersionControl = () => {
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [mergeRequests, setMergeRequests] = useState<MergeRequest[]>([]);
  const [tags, setTags] = useState<TagInfo[]>([]);
  const [currentBranch, setCurrentBranch] = useState<Branch | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch versions
  const fetchVersions = useCallback(async (strategyId: string, branch?: string) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      params.append('strategy_id', strategyId);
      if (branch) params.append('branch', branch);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/versions?${params}`);
      const data = await response.json();
      
      setVersions(data.map((version: any) => ({
        ...version,
        timestamp: new Date(version.timestamp)
      })));
    } catch (err) {
      console.error('Failed to fetch versions:', err);
      setError('Failed to fetch versions');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch branches
  const fetchBranches = useCallback(async (strategyId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/branches?strategy_id=${strategyId}`);
      const data = await response.json();
      
      setBranches(data.map((branch: any) => ({
        ...branch,
        createdAt: new Date(branch.createdAt),
        lastModified: new Date(branch.lastModified)
      })));
    } catch (err) {
      console.error('Failed to fetch branches:', err);
      setError('Failed to fetch branches');
    }
  }, []);

  // Fetch merge requests
  const fetchMergeRequests = useCallback(async (strategyId: string, status?: string) => {
    try {
      const params = new URLSearchParams();
      params.append('strategy_id', strategyId);
      if (status) params.append('status', status);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/merge-requests?${params}`);
      const data = await response.json();
      
      setMergeRequests(data.map((mr: any) => ({
        ...mr,
        requestedAt: new Date(mr.requestedAt),
        approvals: mr.approvals.map((approval: any) => ({
          ...approval,
          approvedAt: new Date(approval.approvedAt)
        }))
      })));
    } catch (err) {
      console.error('Failed to fetch merge requests:', err);
      setError('Failed to fetch merge requests');
    }
  }, []);

  // Fetch tags
  const fetchTags = useCallback(async (strategyId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/tags?strategy_id=${strategyId}`);
      const data = await response.json();
      
      setTags(data.map((tag: any) => ({
        ...tag,
        taggedAt: new Date(tag.taggedAt)
      })));
    } catch (err) {
      console.error('Failed to fetch tags:', err);
      setError('Failed to fetch tags');
    }
  }, []);

  // Create new version
  const createVersion = useCallback(async (
    strategyId: string,
    strategyCode: string,
    strategyConfig: Record<string, any>,
    message: string,
    branch: string = 'main',
    author: string = 'user'
  ): Promise<VersionInfo | null> => {
    try {
      setLoading(true);
      
      const versionData = {
        strategy_id: strategyId,
        strategy_code: strategyCode,
        strategy_config: strategyConfig,
        message,
        branch,
        author
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/versions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(versionData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create version: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Version created successfully`);
      
      await fetchVersions(strategyId, branch);
      
      return {
        ...result,
        timestamp: new Date(result.timestamp)
      };
    } catch (err) {
      console.error('Failed to create version:', err);
      message.error(`Failed to create version: ${err}`);
      setError(`Failed to create version: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchVersions]);

  // Create new branch
  const createBranch = useCallback(async (
    strategyId: string,
    branchName: string,
    baseVersion?: string,
    description?: string,
    createdBy: string = 'user'
  ): Promise<Branch | null> => {
    try {
      setLoading(true);
      
      const branchData = {
        strategy_id: strategyId,
        name: branchName,
        base_version: baseVersion,
        description,
        created_by: createdBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/branches`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(branchData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create branch: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Branch created successfully`);
      
      await fetchBranches(strategyId);
      
      return {
        ...result,
        createdAt: new Date(result.createdAt),
        lastModified: new Date(result.lastModified)
      };
    } catch (err) {
      console.error('Failed to create branch:', err);
      message.error(`Failed to create branch: ${err}`);
      setError(`Failed to create branch: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchBranches]);

  // Switch branch
  const switchBranch = useCallback(async (
    strategyId: string,
    branchName: string
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/switch-branch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: strategyId,
          branch_name: branchName
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to switch branch: ${response.statusText}`);
      }

      message.success(`Switched to branch ${branchName}`);
      
      await fetchBranches(strategyId);
      await fetchVersions(strategyId, branchName);
      
      const branch = branches.find(b => b.name === branchName);
      if (branch) {
        setCurrentBranch(branch);
      }
      
      return true;
    } catch (err) {
      console.error('Failed to switch branch:', err);
      message.error(`Failed to switch branch: ${err}`);
      setError(`Failed to switch branch: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchBranches, fetchVersions, branches]);

  // Create merge request
  const createMergeRequest = useCallback(async (
    strategyId: string,
    sourceBranch: string,
    targetBranch: string,
    title: string,
    description: string,
    reviewers: string[] = [],
    requestedBy: string = 'user'
  ): Promise<MergeRequest | null> => {
    try {
      setLoading(true);
      
      const mergeData = {
        strategy_id: strategyId,
        source_branch: sourceBranch,
        target_branch: targetBranch,
        title,
        description,
        reviewers,
        requested_by: requestedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/merge-requests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mergeData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create merge request: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Merge request created successfully`);
      
      await fetchMergeRequests(strategyId);
      
      return {
        ...result,
        requestedAt: new Date(result.requestedAt),
        approvals: result.approvals.map((approval: any) => ({
          ...approval,
          approvedAt: new Date(approval.approvedAt)
        }))
      };
    } catch (err) {
      console.error('Failed to create merge request:', err);
      message.error(`Failed to create merge request: ${err}`);
      setError(`Failed to create merge request: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchMergeRequests]);

  // Approve merge request
  const approveMergeRequest = useCallback(async (
    mergeId: string,
    reviewer: string,
    comments?: string
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/merge-requests/${mergeId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviewer,
          comments
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to approve merge request: ${response.statusText}`);
      }

      message.success(`Merge request approved successfully`);
      
      const mergeRequest = mergeRequests.find(mr => mr.mergeId === mergeId);
      if (mergeRequest) {
        await fetchMergeRequests(mergeRequest.strategyId);
      }
      
      return true;
    } catch (err) {
      console.error('Failed to approve merge request:', err);
      message.error(`Failed to approve merge request: ${err}`);
      setError(`Failed to approve merge request: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchMergeRequests, mergeRequests]);

  // Merge branches
  const mergeBranches = useCallback(async (
    mergeId: string
  ): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/merge-requests/${mergeId}/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to merge branches: ${response.statusText}`);
      }

      message.success(`Branches merged successfully`);
      
      const mergeRequest = mergeRequests.find(mr => mr.mergeId === mergeId);
      if (mergeRequest) {
        await fetchMergeRequests(mergeRequest.strategyId);
        await fetchVersions(mergeRequest.strategyId, mergeRequest.targetBranch);
        await fetchBranches(mergeRequest.strategyId);
      }
      
      return true;
    } catch (err) {
      console.error('Failed to merge branches:', err);
      message.error(`Failed to merge branches: ${err}`);
      setError(`Failed to merge branches: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchMergeRequests, fetchVersions, fetchBranches, mergeRequests]);

  // Compare versions
  const compareVersions = useCallback(async (
    strategyId: string,
    baseVersion: string,
    compareVersion: string
  ): Promise<ComparisonResult | null> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: strategyId,
          base_version: baseVersion,
          compare_version: compareVersion
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to compare versions: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      console.error('Failed to compare versions:', err);
      message.error(`Failed to compare versions: ${err}`);
      setError(`Failed to compare versions: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Create tag
  const createTag = useCallback(async (
    versionId: string,
    tagName: string,
    description?: string,
    isRelease: boolean = false,
    taggedBy: string = 'user'
  ): Promise<TagInfo | null> => {
    try {
      setLoading(true);
      
      const tagData = {
        version_id: versionId,
        name: tagName,
        description,
        is_release: isRelease,
        tagged_by: taggedBy
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/version-control/tags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tagData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create tag: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Tag created successfully`);
      
      const version = versions.find(v => v.versionId === versionId);
      if (version) {
        await fetchTags(version.strategyId);
      }
      
      return {
        ...result,
        taggedAt: new Date(result.taggedAt)
      };
    } catch (err) {
      console.error('Failed to create tag:', err);
      message.error(`Failed to create tag: ${err}`);
      setError(`Failed to create tag: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchTags, versions]);

  // Get version by ID
  const getVersion = useCallback((versionId: string): VersionInfo | null => {
    return versions.find(v => v.versionId === versionId) || null;
  }, [versions]);

  // Get branch by name
  const getBranch = useCallback((branchName: string): Branch | null => {
    return branches.find(b => b.name === branchName) || null;
  }, [branches]);

  // Get merge request by ID
  const getMergeRequest = useCallback((mergeId: string): MergeRequest | null => {
    return mergeRequests.find(mr => mr.mergeId === mergeId) || null;
  }, [mergeRequests]);

  // Get latest version in branch
  const getLatestVersion = useCallback((strategyId: string, branch: string = 'main'): VersionInfo | null => {
    const branchVersions = versions.filter(v => v.strategyId === strategyId && v.branch === branch);
    return branchVersions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0] || null;
  }, [versions]);

  return {
    // State
    versions,
    branches,
    mergeRequests,
    tags,
    currentBranch,
    loading,
    error,

    // Actions
    createVersion,
    createBranch,
    switchBranch,
    createMergeRequest,
    approveMergeRequest,
    mergeBranches,
    compareVersions,
    createTag,

    // Queries
    getVersion,
    getBranch,
    getMergeRequest,
    getLatestVersion,

    // Data fetching
    fetchVersions,
    fetchBranches,
    fetchMergeRequests,
    fetchTags,

    // State setters
    setCurrentBranch
  };
};