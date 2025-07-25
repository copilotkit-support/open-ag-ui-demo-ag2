interface SectionTitleProps {
  title: string
}

export function SectionTitle({ title }: SectionTitleProps) {
  return (
    <div className="border-b border-border pb-1">
      <h2 className="text-lg font-semibold text-foreground font-['Roobert']">{title}</h2>
    </div>
  )
}
