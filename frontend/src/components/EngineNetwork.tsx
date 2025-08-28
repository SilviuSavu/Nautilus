import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Grid,
  Chip,
  Alert,
  Switch,
  FormControlLabel,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  List,
  ListItem,
  ListItemText,
  Badge,
  LinearProgress,
} from '@mui/material';
import {
  Timeline,
  Settings,
  Info,
  Refresh,
  Warning,
  Error,
  CheckCircle,
  Speed,
  Group,
  Hub,
} from '@mui/icons-material';
import { 
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Types for engine network data
interface EngineNode {
  id: string;
  name: string;
  type: string;
  status: string;
  properties: {
    port: number;
    roles: string[];
    capabilities: string[];
    performance: {
      response_time_ms: number;
      throughput: number;
      acceleration_factor: number;
    };
    health: {
      status: string;
      uptime_seconds: number;
      load_pct: number;
    };
  };
}

interface EngineEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  strength: number;
  properties: {
    status: string;
    performance_score: number;
    message_count: number;
  };
}

interface NetworkTopology {
  nodes: EngineNode[];
  edges: EngineEdge[];
  clusters: Record<string, string[]>;
  metadata: {
    total_nodes: number;
    total_edges: number;
    generated_at: string;
    engine_types: string[];
  };
}

interface SystemStatus {
  total_engines: number;
  online_engines: number;
  active_partnerships: number;
  active_tasks: number;
  system_health: string;
  last_updated: string;
}

interface LiveMetrics {
  timestamp: string;
  system_status: SystemStatus;
  engine_metrics: Record<string, any>;
  partnership_metrics: Record<string, any>;
  alerts: Array<{
    type: string;
    source: string;
    message: string;
    timestamp: string;
  }>;
}

// Custom node components for React Flow
const EngineNodeComponent = ({ data }: { data: any }) => {
  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
      case 'operational':
      case 'active':
        return '#4caf50';
      case 'degraded':
      case 'warning':
        return '#ff9800';
      case 'critical':
      case 'offline':
      case 'failed':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'engine':
        return <Settings fontSize="small" />;
      case 'messagebus':
        return <Hub fontSize="small" />;
      default:
        return <Settings fontSize="small" />;
    }
  };

  return (
    <Card
      sx={{
        minWidth: 200,
        border: `2px solid ${getStatusColor(data.status)}`,
        backgroundColor: data.type === 'messagebus' ? '#e8f5e8' : '#f5f5f5',
      }}
    >
      <CardContent sx={{ padding: '8px !important' }}>
        <Box display="flex" alignItems="center" gap={1}>
          {getTypeIcon(data.type)}
          <Typography variant="subtitle2" fontWeight="bold">
            {data.name}
          </Typography>
        </Box>
        
        <Typography variant="caption" color="text.secondary">
          Port: {data.properties?.port || 'N/A'}
        </Typography>
        
        <Box mt={1}>
          <Chip 
            label={data.status} 
            size="small" 
            sx={{ 
              backgroundColor: getStatusColor(data.status),
              color: 'white',
              fontSize: '0.7rem'
            }}
          />
        </Box>
        
        {data.properties?.performance && (
          <Box mt={1}>
            <Typography variant="caption" display="block">
              Response: {data.properties.performance.response_time_ms?.toFixed(1)}ms
            </Typography>
            <Typography variant="caption" display="block">
              Load: {data.properties.health?.load_pct?.toFixed(0)}%
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Main EngineNetwork component
const EngineNetwork: React.FC = () => {
  const [topology, setTopology] = useState<NetworkTopology | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // WebSocket connection for real-time updates
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Fetch network topology
  const fetchTopology = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/network/topology');
      if (!response.ok) throw new Error('Failed to fetch topology');
      const data = await response.json();
      setTopology(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  // Fetch system status
  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/system/status');
      if (!response.ok) throw new Error('Failed to fetch system status');
      const data = await response.json();
      
      setLiveMetrics(prev => prev ? {
        ...prev,
        system_status: data
      } : null);
    } catch (err) {
      console.error('Error fetching system status:', err);
    }
  }, []);

  // Initialize WebSocket connection
  useEffect(() => {
    if (!autoRefresh) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/live-metrics`;
    
    const websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };
    
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'metrics_update') {
          setLiveMetrics(data.data);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setWs(null);
      
      // Attempt to reconnect after delay
      if (autoRefresh) {
        setTimeout(() => {
          // Re-trigger effect to reconnect
        }, 5000);
      }
    };
    
    return () => {
      websocket.close();
    };
  }, [autoRefresh]);

  // Initial data fetch
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchTopology(), fetchSystemStatus()]);
      setLoading(false);
    };
    
    loadData();
  }, [fetchTopology, fetchSystemStatus]);

  // Convert topology to React Flow format
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!topology) return { flowNodes: [], flowEdges: [] };

    // Create nodes
    const flowNodes: Node[] = topology.nodes.map((node, index) => {
      const angle = (index / topology.nodes.length) * 2 * Math.PI;
      const radius = Math.min(400, 200 + topology.nodes.length * 10);
      
      return {
        id: node.id,
        type: 'default',
        position: {
          x: Math.cos(angle) * radius + 500,
          y: Math.sin(angle) * radius + 300,
        },
        data: {
          label: <EngineNodeComponent data={node} />,
          ...node,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    });

    // Create edges
    const flowEdges: Edge[] = topology.edges.map((edge) => {
      const getEdgeColor = (type: string, strength: number) => {
        if (type === 'partnership') {
          if (strength > 0.8) return '#4caf50';
          if (strength > 0.5) return '#ff9800';
          return '#f44336';
        }
        return '#9e9e9e';
      };

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
        style: {
          stroke: getEdgeColor(edge.type, edge.strength),
          strokeWidth: Math.max(1, edge.strength * 4),
        },
        label: `${edge.type} (${(edge.strength * 100).toFixed(0)}%)`,
        labelStyle: { fontSize: 10, fill: '#666' },
        animated: edge.properties.status === 'active',
      };
    });

    return { flowNodes, flowEdges };
  }, [topology]);

  // Update React Flow nodes and edges
  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  // Handle node click
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedEngine(node.id);
    setShowDetails(true);
  }, []);

  // Refresh data manually
  const handleRefresh = useCallback(() => {
    fetchTopology();
    fetchSystemStatus();
  }, [fetchTopology, fetchSystemStatus]);

  // Get health status color
  const getHealthColor = (health: string) => {
    switch (health?.toLowerCase()) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'critical': return 'error';
      default: return 'info';
    }
  };

  if (loading) {
    return (
      <Box p={3}>
        <Typography variant="h4" gutterBottom>Engine Network</Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2 }}>Loading engine network data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography variant="h4" gutterBottom>Engine Network</Typography>
        <Alert severity="error">
          Failed to load engine network: {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom display="flex" alignItems="center" gap={1}>
        <Hub />
        Engine Network
      </Typography>

      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                }
                label="Auto Refresh"
              />
            </Grid>
            
            <Grid item>
              <IconButton onClick={handleRefresh} disabled={loading}>
                <Refresh />
              </IconButton>
            </Grid>

            {liveMetrics?.system_status && (
              <>
                <Grid item>
                  <Chip
                    icon={<CheckCircle />}
                    label={`${liveMetrics.system_status.online_engines}/${liveMetrics.system_status.total_engines} Online`}
                    color="success"
                    variant="outlined"
                  />
                </Grid>
                
                <Grid item>
                  <Chip
                    icon={<Group />}
                    label={`${liveMetrics.system_status.active_partnerships} Partnerships`}
                    color="primary"
                    variant="outlined"
                  />
                </Grid>
                
                <Grid item>
                  <Chip
                    icon={<Speed />}
                    label={`${liveMetrics.system_status.active_tasks} Active Tasks`}
                    color="info"
                    variant="outlined"
                  />
                </Grid>
                
                <Grid item>
                  <Chip
                    label={liveMetrics.system_status.system_health}
                    color={getHealthColor(liveMetrics.system_status.system_health)}
                  />
                </Grid>
              </>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Alerts */}
      {liveMetrics?.alerts && liveMetrics.alerts.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardHeader
            title="System Alerts"
            titleTypographyProps={{ variant: 'h6' }}
            avatar={<Warning color="warning" />}
          />
          <CardContent>
            {liveMetrics.alerts.map((alert, index) => (
              <Alert 
                key={index} 
                severity={alert.type === 'warning' ? 'warning' : 'error'}
                sx={{ mb: 1 }}
              >
                <strong>{alert.source}:</strong> {alert.message}
              </Alert>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Network Visualization */}
      <Card sx={{ height: 600 }}>
        <CardHeader
          title="Network Topology"
          titleTypographyProps={{ variant: 'h6' }}
          action={
            <Typography variant="caption" color="text.secondary">
              {topology?.metadata.total_nodes} nodes, {topology?.metadata.total_edges} connections
            </Typography>
          }
        />
        <CardContent sx={{ height: '100%', p: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            fitView
            attributionPosition="top-right"
          >
            <Background />
            <Controls />
            <MiniMap 
              nodeStrokeColor="#333"
              nodeColor="#fff"
              nodeBorderRadius={2}
            />
          </ReactFlow>
        </CardContent>
      </Card>

      {/* Engine Details Dialog */}
      <Dialog 
        open={showDetails} 
        onClose={() => setShowDetails(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Engine Details: {selectedEngine}
        </DialogTitle>
        <DialogContent>
          {selectedEngine && topology && (
            <Box>
              {(() => {
                const engine = topology.nodes.find(n => n.id === selectedEngine);
                if (!engine) return <Typography>Engine not found</Typography>;
                
                return (
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>Basic Information</Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText primary="Name" secondary={engine.name} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Type" secondary={engine.type} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Port" secondary={engine.properties.port} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Status" secondary={engine.status} />
                        </ListItem>
                      </List>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>Performance</Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText 
                            primary="Response Time" 
                            secondary={`${engine.properties.performance?.response_time_ms || 'N/A'}ms`} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Throughput" 
                            secondary={engine.properties.performance?.throughput || 'N/A'} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="CPU Load" 
                            secondary={`${engine.properties.health?.load_pct || 0}%`} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Uptime" 
                            secondary={`${Math.floor((engine.properties.health?.uptime_seconds || 0) / 3600)}h`} 
                          />
                        </ListItem>
                      </List>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom>Capabilities</Typography>
                      <Box display="flex" flexWrap="wrap" gap={1}>
                        {engine.properties.capabilities?.map((cap, index) => (
                          <Chip key={index} label={cap} size="small" />
                        ))}
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom>Roles</Typography>
                      <Box display="flex" flexWrap="wrap" gap={1}>
                        {engine.properties.roles?.map((role, index) => (
                          <Chip key={index} label={role} size="small" color="primary" />
                        ))}
                      </Box>
                    </Grid>
                  </Grid>
                );
              })()}
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default EngineNetwork;