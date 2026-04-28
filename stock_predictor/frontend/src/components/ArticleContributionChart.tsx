import type { Article } from '../types'

function contribution(article: Article): number {
  return article.relevance_score * article.finbert.score
}

export function ArticleContributionChart({ articles }: { articles: Article[] }) {
  const maxAbs = Math.max(0.0001, ...articles.map((a) => Math.abs(contribution(a))))

  return (
    <section className="card">
      <h3>Article Contribution View</h3>
      {!articles.length && <p className="subtitle">No contribution data available.</p>}
      {articles.map((article, i) => {
        const c = contribution(article)
        const width = `${(Math.abs(c) / maxAbs) * 100}%`
        return (
          <div key={article.uuid || `${article.url}-${i}`} className="contrib-row">
            <p className="contrib-title">
              #{i + 1} {article.title}
            </p>
            <p className="subtitle">
              relevance={article.relevance_score.toFixed(4)} | finbert={article.finbert.score.toFixed(4)} | marketaux=
              {(article.entity_sentiment_score ?? 0).toFixed(4)} | weighted={c.toFixed(4)}
            </p>
            <div className="bar-wrap">
              <div className={`bar ${c >= 0 ? 'positive' : 'negative'}`} style={{ width }} />
            </div>
          </div>
        )
      })}
    </section>
  )
}
