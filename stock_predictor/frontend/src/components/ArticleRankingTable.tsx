import type { Article } from '../types'

export function ArticleRankingTable({ articles }: { articles: Article[] }) {
  return (
    <section className="card">
      <h3>Article Selection (Top Ranked)</h3>
      {!articles.length && <p className="subtitle">No relevant articles found.</p>}
      {articles.map((article, index) => (
        <article key={article.uuid || `${article.url}-${index}`} className="article">
          <p className="meta-row">
            <span className="rank-chip">#{index + 1}</span>
            <span>relevance: {article.relevance_score.toFixed(4)}</span>
            <span>source: {article.source || 'unknown'}</span>
          </p>
          <a href={article.url} target="_blank" rel="noreferrer">
            {article.title || 'Untitled article'}
          </a>
          <p className="subtitle">{new Date(article.published_at).toLocaleDateString()}</p>
          {article.highlights?.[0]?.highlight && <p className="muted">{article.highlights[0].highlight}</p>}
        </article>
      ))}
    </section>
  )
}
