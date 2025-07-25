"use client"

import type React from "react"
import { CopilotChat } from "@copilotkit/react-ui"


interface PromptPanelProps {
  availableCash: number
}



export function PromptPanel({ availableCash }: PromptPanelProps) {


  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }



  return (
    <div className="h-full flex flex-col bg-card">
      {/* Header */}
      <div className="p-4 border-b border-border bg-muted">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xl">🪁</span>
          <div>
            <h1 className="text-lg font-semibold text-foreground font-['Roobert']">Portfolio Chat</h1>
            <div className="inline-block px-2 py-0.5 bg-accent/20 text-foreground text-xs font-semibold uppercase rounded">
              PRO
            </div>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">Interact with the AG2 AI agent for portfolio visualization and analysis</p>

        {/* Available Cash Display */}
        <div className="mt-3 p-2 bg-accent/10 rounded-lg">
          <div className="text-xs text-muted-foreground font-medium">Available Cash</div>
          <div className="text-sm font-semibold text-foreground font-['Roobert']">{formatCurrency(availableCash)}</div>
        </div>
      </div>
      <div
        style={
          {
            "--copilot-kit-background-color": "#2a2a2a",
            "--copilot-kit-secondary-color": "#808080",
            "--copilot-kit-input-background-color" : "#3a3a3a",
            "--copilot-kit-separator-color": "#3a3a3a",
            "--copilot-kit-primary-color": "#FFFFFF",
            "--copilot-kit-contrast-color": "#000000",
            "--copilot-kit-secondary-contrast-color": "#808080",
          } as any
        }
      > <CopilotChat className="h-[77vh] hide-scrollbar" labels={
        {
          initial: `I am a AG2 AI agent designed to analyze investment opportunities and track stock performance over time. How can I help you with your investment query? For example, you can ask me to analyze a stock like "Invest in Apple with 10k dollars since Jan 2023". \n\nNote: The AI agent has access to stock data from the past 4 years only`
        }
      } /></div>

    </div >
  )
}
