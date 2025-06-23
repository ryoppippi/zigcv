#set document(title: "ccusage - Claude Code Usage Analysis Tool")
#set page(paper: "a4", margin: 2cm)
#set heading(numbering: "1.")

= ccusage - Claude Code Usage Analysis Tool

#link("https://ccusage.com/")[ccusage] is a CLI tool for analyzing Claude Code usage from local JSONL files, created by #link("https://github.com/ryoppippi")[\@ryoppippi].

== Overview

ccusage provides comprehensive analysis of your Claude Code interactions, including:

- Token usage tracking
- Cost estimation in USD
- Session-based analysis
- Real-time monitoring capabilities
- Model-specific tracking (Opus, Sonnet)

== Key Features

=== Reporting Capabilities

- *Daily Reports*: View token usage and costs aggregated by date
- *Monthly Reports*: View token usage and costs aggregated by month
- *Session Reports*: View usage grouped by conversation sessions
- *5-Hour Blocks Report*: Track usage within Claude's billing windows with active block monitoring
- *Live Monitoring*: Real-time dashboard showing active session progress, token burn rate, and cost projections

=== Technical Features

- 📊 Responsive terminal tables with smart column sizing
- 📄 JSON output support for programmatic use
- 💰 Cost tracking with USD estimates
- 🤖 Per-model usage breakdown
- 📈 Cache token tracking (creation and reads)
- 🌐 Offline mode with pre-cached pricing data
- 🔌 MCP (Model Context Protocol) integration

== Installation

=== Quick Start
```bash
npx ccusage@latest
```

=== Global Installation
```bash
npm install -g ccusage
```

=== Package Manager
```bash
npm i ccusage
```

== Usage Examples

=== Basic Commands

```bash
# Show daily report (default)
ccusage

# Daily token usage and costs
ccusage daily

# Monthly aggregated report
ccusage monthly

# Usage by conversation session
ccusage session

# 5-hour billing windows
ccusage blocks

# Real-time usage dashboard
ccusage blocks --live
```

=== Command Options

All commands support various options:
- Date range filtering
- Model-specific filtering
- JSON output format
- Custom configuration paths

== Additional Tools

=== Raycast Extension
A Raycast extension is available that provides real-time monitoring of Claude Code usage statistics using the ccusage CLI tool.

=== Related Projects
- *Claude Code Usage Monitor*: A terminal monitoring tool for real-time Claude AI token usage tracking with burn rate predictions

== Resources

- *Official Website*: #link("https://ccusage.com/")
- *GitHub Repository*: #link("https://github.com/ryoppippi/ccusage")
- *npm Package*: #link("https://www.npmjs.com/package/ccusage")
- *Documentation*: #link("https://ccusage.com/")

== License

MIT License

== Version Information

Latest version: 15.1.0 (as of documentation creation)

The tool is actively maintained with frequent updates and improvements.