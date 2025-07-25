interface Insight {
  title: string
  description: string
  emoji: string
}

interface InsightCardComponentProps {
  insight: Insight
  type: "bull" | "bear"
}

export function InsightCardComponent({ insight, type }: InsightCardComponentProps) {
  const getTypeStyles = () => {
    switch (type) {
      case "bull":
        return "border-l-4 border-l-accent bg-green-500/10"
      case "bear":
        return "border-l-4 border-l-red-400 bg-red-500/10"
      default:
        return "border-l-4 border-l-border"
    }
  }

  return (
    <div className={`bg-card border border-border rounded-xl p-3 ${getTypeStyles()}`}>
      <div className="flex items-start gap-2">
        <span className="text-lg">{insight.emoji}</span>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-foreground font-['Roobert'] mb-1">{insight.title}</h3>
          <p className="text-xs text-muted-foreground font-['Plus_Jakarta_Sans'] leading-relaxed">{insight.description}</p>
        </div>
      </div>
    </div>
  )
}
