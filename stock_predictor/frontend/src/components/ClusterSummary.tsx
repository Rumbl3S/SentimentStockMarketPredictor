import type { PredictResponse } from '../types'

export function ClusterSummary({ clusterInfo }: { clusterInfo: PredictResponse['cluster_info'] }) {
  if (!clusterInfo?.enabled) {
    return null
  }

  return (
    <section className="card">
      <h3>Optional Clustering Summary</h3>
      <p className="subtitle">Query cluster: {clusterInfo.query_cluster}</p>
      {Object.entries(clusterInfo.clusters).map(([clusterId, cluster]) => (
        <div key={clusterId} className="cluster">
          <p className="contrib-title">
            Cluster {clusterId}: {cluster.size} articles
          </p>
          <p className="subtitle"><strong>Top terms:</strong> {cluster.top_terms.join(', ') || 'none'}</p>
          <p className="subtitle"><strong>Top tickers:</strong> {cluster.top_tickers.map(([t, count]) => `${t}:${count}`).join(', ') || 'none'}</p>
        </div>
      ))}
    </section>
  )
}
