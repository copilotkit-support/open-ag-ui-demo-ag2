from ag_ui.core import StateDeltaEvent, EventType
from ag_ui.core.types import AssistantMessage, ToolMessage
import yfinance as yf
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import (
    AgentNameTarget,
    ReplyResult,
    ContextVariables,
)
from prompts import insights_prompt
import os
import numpy as np
import pandas as pd
from typing import List
from openai import OpenAI

load_dotenv()
from dataclasses import dataclass, field
from typing import List



@dataclass
class Insight:
    title: str
    description: str
    emoji: str

# Configure the LLM
llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.2,
    parallel_tool_calls = False
)

generate_insights_tool = {
    "type": "function",
    "function": {
        "name": "generate_insights",
        "description": "Generate positive (bull) and negative (bear) insights for a stock or portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "bullInsights": {
                    "type": "array",
                    "description": "A list of positive insights (bull case) for the stock or portfolio.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short title for the positive insight.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the positive insight.",
                            },
                            "emoji": {
                                "type": "string",
                                "description": "Emoji representing the positive insight.",
                            },
                        },
                        "required": ["title", "description", "emoji"],
                    },
                },
                "bearInsights": {
                    "type": "array",
                    "description": "A list of negative insights (bear case) for the stock or portfolio.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short title for the negative insight.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the negative insight.",
                            },
                            "emoji": {
                                "type": "string",
                                "description": "Emoji representing the negative insight.",
                            },
                        },
                        "required": ["title", "description", "emoji"],
                    },
                },
            },
            "required": ["bullInsights", "bearInsights"],
        },
    }
}

async def extract_relevant_data_from_user_prompt(
    context_variables: ContextVariables,
    ticker_symbols: list[str],
    investment_date: str,
    amount_of_dollars_to_be_invested: list[int],
    to_be_added_in_portfolio: bool,
):
    context_variables.set('tool_logs', [])
    tool_log_id = str(uuid.uuid4())
    context_variables.set('be_arguments', {
        "ticker_symbols": ticker_symbols,
        "investment_date": investment_date,
        "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
        "to_be_added_in_portfolio": to_be_added_in_portfolio,
    })
    context_variables.get('sample_function')(10)
    context_variables.get('tool_logs').append(
        {
            "id": tool_log_id,
            "message": "Analyzing user query",
            "status": "processing",
        }
    )
    context_variables.set('investment_portfolio', json.dumps(
        [
            {
                "ticker": ticker,
                "amount": amount_of_dollars_to_be_invested[index],
            }
            for index, ticker in enumerate(ticker_symbols)
        ]
    ))
    # context_variables.data["be_arguments"] = {
    #     "ticker_symbols": ticker_symbols,
    #     "investment_date": investment_date,
    #     "amount_of_dollars_to_be_invested": amount_of_dollars_to_be_invested,
    #     "to_be_added_in_portfolio": to_be_added_in_portfolio,
    # }
    # print(context_variables)
    index = len(context_variables.get('tool_logs')) - 1
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)
    context_variables.get('tool_logs').append(
        {
            "id": tool_log_id,
            "message": "Gathering stock data",
            "status": "processing",
        }
    )
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Gathering stock data",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )    
    await asyncio.sleep(0)
    return ReplyResult(
        message="User's query has been processed and relevant data has been extracted",
        context_variables=context_variables,
        target=AgentNameTarget("stock_data_bot"),
    )

async def gather_stock_data(context_variables: ContextVariables, tikers : list[str]):
    tool_log_id = str(uuid.uuid4())
    tickers = tikers
    print ('DEBUG: tickers in gather_stock_data', tickers)
    investment_date = context_variables.get('be_arguments')["investment_date"]
    current_year = datetime.now().year
    if current_year - int(investment_date[:4]) > 4:
        print("investment date is more than 4 years ago")
        investment_date = f"{current_year - 4}-01-01"
    if current_year - int(investment_date[:4]) == 0:
        history_period = "1y"
    else:
        history_period = f"{current_year - int(investment_date[:4])}y"

    data = yf.download(
        tickers,
        period=history_period,
        interval="3mo",
        start=investment_date,
        end=datetime.today().strftime("%Y-%m-%d"),
    )
    context_variables.set('be_stock_data', data["Close"])
    # print(context_variables.data,"HERERERERER")
    index = len(context_variables.get('tool_logs')) - 1
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(2)
    context_variables.get('tool_logs').append(
        {
            "id": tool_log_id,
            "message": "Allocating cash",
            "status": "processing",
        }
    )
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Allocating cash",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )    
    await asyncio.sleep(2)
    return ReplyResult(
        message="Stock data had been gathered successfully",
        context_variables=context_variables,
        target= AgentNameTarget('cash_allocation_bot')
    )

async def allocate_cash(context_variables : ContextVariables, amount_of_dollars_to_be_invested : list[int]):
    stock_data = context_variables.get('be_stock_data')  # DataFrame: index=date, columns=tickers
    args = context_variables.get('be_arguments')
    tickers = args["ticker_symbols"]
    print ('DEBUG: tickers in allocate_cash', tickers)
    investment_date = args["investment_date"]
    amounts = amount_of_dollars_to_be_invested  # list, one per ticker
    interval = args.get("interval_of_investment", "single_shot")

    # Use state['available_cash'] as a single integer (total wallet cash)
    if context_variables.get('available_cash'):
        total_cash = context_variables.get('available_cash')
    else:
        total_cash = sum(amounts)
    holdings = {ticker: 0.0 for ticker in tickers}
    investment_log = []
    add_funds_needed = False
    add_funds_dates = []

    # Ensure DataFrame is sorted by date
    stock_data = stock_data.sort_index()

    if interval == "single_shot":
        # Buy all shares at the first available date using allocated money for each ticker
        first_date = stock_data.index[0]
        row = stock_data.loc[first_date]
        for idx, ticker in enumerate(tickers):
            price = row[ticker]
            if np.isnan(price):
                investment_log.append(
                    f"{first_date.date()}: No price data for {ticker}, could not invest."
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, amounts[idx])
                )
                continue
            allocated = amounts[idx]
            if total_cash >= allocated and allocated >= price:
                shares_to_buy = allocated // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    holdings[ticker] += shares_to_buy
                    total_cash -= cost
                    investment_log.append(
                        f"{first_date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                    )
                else:
                    investment_log.append(
                        f"{first_date.date()}: Not enough allocated cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}"
                    )
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(first_date.date()), ticker, price, allocated)
                    )
            else:
                investment_log.append(
                    f"{first_date.date()}: Not enough total cash to buy {ticker} at ${price:.2f}. Allocated: ${allocated:.2f}, Available: ${total_cash:.2f}"
                )
                add_funds_needed = True
                add_funds_dates.append(
                    (str(first_date.date()), ticker, price, total_cash)
                )
        # No further purchases on subsequent dates
    else:
        # DCA or other interval logic (previous logic)
        for date, row in stock_data.iterrows():
            for i, ticker in enumerate(tickers):
                price = row[ticker]
                if np.isnan(price):
                    continue  # skip if price is NaN
                # Invest as much as possible for this ticker at this date
                if total_cash >= price:
                    shares_to_buy = total_cash // price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        holdings[ticker] += shares_to_buy
                        total_cash -= cost
                        investment_log.append(
                            f"{date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                        )
                else:
                    add_funds_needed = True
                    add_funds_dates.append(
                        (str(date.date()), ticker, price, total_cash)
                    )
                    investment_log.append(
                        f"{date.date()}: Not enough cash to buy {ticker} at ${price:.2f}. Available: ${total_cash:.2f}. Please add more funds."
                    )

    # Calculate final value and new summary fields
    final_prices = stock_data.iloc[-1]
    total_value = 0.0
    returns = {}
    total_invested_per_stock = {}
    percent_allocation_per_stock = {}
    percent_return_per_stock = {}
    total_invested = 0.0
    for idx, ticker in enumerate(tickers):
        # Calculate how much was actually invested in this stock
        if interval == "single_shot":
            # Only one purchase at first date
            first_date = stock_data.index[0]
            price = stock_data.loc[first_date][ticker]
            shares_bought = holdings[ticker]
            invested = shares_bought * price
        else:
            # Sum all purchases from the log
            invested = 0.0
            for log in investment_log:
                if f"shares of {ticker}" in log and "Bought" in log:
                    # Extract cost from log string
                    try:
                        cost_str = log.split("(cost: $")[-1].split(")")[0]
                        invested += float(cost_str)
                    except Exception:
                        pass
        total_invested_per_stock[ticker] = invested
        total_invested += invested
    # Now calculate percent allocation and percent return
    for ticker in tickers:
        invested = total_invested_per_stock[ticker]
        holding_value = holdings[ticker] * final_prices[ticker]
        returns[ticker] = holding_value - invested
        total_value += holding_value
        percent_allocation_per_stock[ticker] = (
            (invested / total_invested * 100) if total_invested > 0 else 0.0
        )
        percent_return_per_stock[ticker] = (
            ((holding_value - invested) / invested * 100) if invested > 0 else 0.0
        )
    total_value += total_cash  # Add remaining cash to total value

    # Store results in state
    context_variables.set('investment_summary', {
        "holdings": holdings,
        "final_prices": final_prices.to_dict(),
        "cash": total_cash,
        "returns": returns,
        "total_value": total_value,
        "investment_log": investment_log,
        "add_funds_needed": add_funds_needed,
        "add_funds_dates": add_funds_dates,
        "total_invested_per_stock": total_invested_per_stock,
        "percent_allocation_per_stock": percent_allocation_per_stock,
        "percent_return_per_stock": percent_return_per_stock,
    })
    context_variables.set('available_cash', total_cash)  # Update available cash in state

    # --- Portfolio vs SPY performanceData logic ---
    # Get SPY prices for the same dates
    spy_ticker = "SPY"
    spy_prices = None
    try:
        spy_prices = yf.download(
            spy_ticker,
            period=f"{len(stock_data)//4}y" if len(stock_data) > 4 else "1y",
            interval="3mo",
            start=stock_data.index[0],
            end=stock_data.index[-1],
        )["Close"]
        # Align SPY prices to stock_data dates
        spy_prices = spy_prices.reindex(stock_data.index, method="ffill")
    except Exception as e:
        print("Error fetching SPY data:", e)
        spy_prices = pd.Series([None] * len(stock_data), index=stock_data.index)

    # Simulate investing the same total_invested in SPY
    spy_shares = 0.0
    spy_cash = total_invested
    spy_invested = 0.0
    spy_investment_log = []
    if interval == "single_shot":
        first_date = stock_data.index[0]
        spy_price = spy_prices.loc[first_date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        if not pd.isna(spy_price):
            spy_shares = spy_cash // spy_price
            spy_invested = spy_shares * spy_price
            spy_cash -= spy_invested
            spy_investment_log.append(
                f"{first_date.date()}: Bought {spy_shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${spy_invested:.2f})"
            )
    else:
        # DCA: invest equal portions at each date
        dca_amount = total_invested / len(stock_data)
        for date in stock_data.index:
            spy_price = spy_prices.loc[date]
            if isinstance(spy_price, pd.Series):
                spy_price = spy_price.iloc[0]
            if not pd.isna(spy_price):
                shares = dca_amount // spy_price
                cost = shares * spy_price
                spy_shares += shares
                spy_cash -= cost
                spy_invested += cost
                spy_investment_log.append(
                    f"{date.date()}: Bought {shares:.2f} shares of SPY at ${spy_price:.2f} (cost: ${cost:.2f})"
                )

    # Build performanceData array
    performanceData = []
    running_holdings = holdings.copy()
    running_cash = total_cash
    for date in stock_data.index:
        # Portfolio value: sum of shares * price at this date + cash
        port_value = (
            sum(
                running_holdings[t] * stock_data.loc[date][t]
                for t in tickers
                if not pd.isna(stock_data.loc[date][t])
            )
            # + running_cash
        )
        # SPY value: shares * price + cash
        spy_price = spy_prices.loc[date]
        if isinstance(spy_price, pd.Series):
            spy_price = spy_price.iloc[0]
        spy_val = spy_shares * spy_price + spy_cash if not pd.isna(spy_price) else None
        performanceData.append(
            {
                "date": str(date.date()),
                "portfolio": float(port_value) if port_value is not None else None,
                "spy": float(spy_val) if spy_val is not None else None,
            }
        )

    context_variables.get('investment_summary')['performanceData'] = performanceData
    # --- End performanceData logic ---

    # Compose summary message
    if add_funds_needed:
        msg = "Some investments could not be made due to insufficient funds. Please add more funds to your wallet.\n"
        for d, t, p, c in add_funds_dates:
            msg += (
                f"On {d}, not enough cash for {t}: price ${p:.2f}, available ${c:.2f}\n"
            )
    else:
        msg = "All investments were made successfully.\n"
    msg += f"\nFinal portfolio value: ${total_value:.2f}\n"
    msg += "Returns by ticker (percent and $):\n"
    for ticker in tickers:
        percent = percent_return_per_stock[ticker]
        abs_return = returns[ticker]
        msg += f"{ticker}: {percent:.2f}% (${abs_return:.2f})\n"

    # context_variables.data["messages"].append(
    #     ToolMessage(
    #         role="tool",
    #         id=str(uuid.uuid4()),
    #         content="The relevant details had been extracted",
    #         tool_call_id=context_variables.data["messages"][-1].tool_calls[0].id,
    #     )
    # )

    context_variables.get('messages').append(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "render_standard_charts_and_table",
                        "arguments": json.dumps(
                            {"investment_summary": context_variables.get('investment_summary')}
                        ),
                    },
                }
            ],
            id=str(uuid.uuid4()),
        )
    )
    print(context_variables.get('investment_summary'), "datatatat")
    index = len(context_variables.get('tool_logs')) - 1
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)
    tool_log_id = str(uuid.uuid4())
    context_variables.get('tool_logs').append(
        {
            "id": tool_log_id,
            "message": "Generating insights",
            "status": "processing",
        }
    )
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "add",
                    "path": "/tool_logs/-",
                    "value": {
                        "message": "Generating insights",
                        "status": "processing",
                        "id": tool_log_id,
                    },
                }
            ],
        )
    )    
    await asyncio.sleep(0)
    return ReplyResult(
        message="Cash had been allocated successfully and returns had been calculated",
        context_variables=context_variables,
        target=AgentNameTarget("insights_bot")
    )


async def generate_insights(context_variables: ContextVariables, tickers : list[str]):
    args = context_variables.get('be_arguments')
    print ('DEBUG: tickers in generate_insights', tickers)
    investment_date = args.get("investment_date", '')
    model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[  
            {"role": "system", "content": insights_prompt},
            {"role": "user", "content": json.dumps(tickers)}
        ],
        tools=[{
    "type": "function",
    "function": {
        "name": "generate_insights",
        "description": "Generate positive (bull) and negative (bear) insights for a stock or portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "bullInsights": {
                    "type": "array",
                    "description": "A list of positive insights (bull case) for the stock or portfolio.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short title for the positive insight.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the positive insight.",
                            },
                            "emoji": {
                                "type": "string",
                                "description": "Emoji representing the positive insight.",
                            },
                        },
                        "required": ["title", "description", "emoji"],
                    },
                },
                "bearInsights": {
                    "type": "array",
                    "description": "A list of negative insights (bear case) for the stock or portfolio.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short title for the negative insight.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the negative insight.",
                            },
                            "emoji": {
                                "type": "string",
                                "description": "Emoji representing the negative insight.",
                            },
                        },
                        "required": ["title", "description", "emoji"],
                    },
                },
            },
            "required": ["bullInsights", "bearInsights"],
        },
    }
}]
    )

    args_dict = json.loads(context_variables.get('messages')[-1].tool_calls[0].function.arguments)

    # Step 2: Add the insights key
    args_dict["insights"] = {
        "bullInsights": json.loads(response.choices[0].message.tool_calls[0].function.arguments)['bullInsights'],
        "bearInsights": json.loads(response.choices[0].message.tool_calls[0].function.arguments)["bearInsights"]
    }
    args_dict["investment_portfolio"] = context_variables.get('investment_portfolio')
    args_dict["investment_date"] = investment_date
    # Step 3: Convert back to string
    context_variables.get('messages')[-1].tool_calls[0].function.arguments = json.dumps(args_dict)
    # print(context_variables.data, "insights")
    index = len(context_variables.get('tool_logs')) - 1
    context_variables.get('emitEvent')(
        StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                {
                    "op": "replace",
                    "path": f"/tool_logs/{index}/status",
                    "value": "completed",
                }
            ],
        )
    )
    await asyncio.sleep(0)
    return ReplyResult(
        message="The Insights for the stocks had been generated successfully",
        context_variables=context_variables,
    )

with llm_config:
    chat_bot = ConversableAgent(
        name="chat_bot",
        system_message="""You are the chat processing agent in this pipeline.

        Your specific role is to process the user's query for his Investment stock portfolio.
        Focus on:
        - Investment tickers array. 
        - If dates are not provided, Take the investment date as 2022-01-01 as default.

        You must always use the extract_relevant_data_from_user_prompt function to extract relevant data from the user query.
        When using the extract_relevant_data_from_user_prompt function, you must always call it only once with multiple tickers in an array. Strictly follow the example format below:
        EXAMPLE FORMAT:
        ticker_symbols = ["AAPL", "MSFT", "GOOG"]
        investment_date = "2022-01-01"
        amount_of_dollars_to_be_invested = [10000, 15000, 20000]
        to_be_added_in_portfolio = True
        
        NOTE: 
        - User will ask you to perform investment queries. Along with that he will provide you with portfolio details. It will contain the various tickers and their amounts. Using this information, you must call the extract_relevant_data_from_user_prompt function to extract the relevant data from the user query along with the portfolio details too.
        - When user asks "Make investments in Nvidia worth 13k dollars  PORTFOLIO DETAILS : [{"ticker": "AAPL", "amount": 15000}, {"ticker": "MSFT", "amount": 20000}]. INVESTMENT DATE : 2022-01-01", you should call the extract_relevant_data_from_user_prompt function with the following arguments:
        ticker_symbols = ["AAPL", "MSFT", "NVDA"]
        investment_date = "2022-01-01"
        amount_of_dollars_to_be_invested = [10000, 15000, 13000]
        to_be_added_in_portfolio = True
        - When user asks "Replace investments of Apple with Nvidia worth 13k dollars  PORTFOLIO DETAILS : [{"ticker": "AAPL", "amount": 15000}, {"ticker": "MSFT", "amount": 20000}]. INVESTMENT DATE : 2022-01-01", you should call the extract_relevant_data_from_user_prompt function with the following arguments:
        ticker_symbols = ["NVDA", "MSFT"]
        amount_of_dollars_to_be_invested = [13000, 20000]
        to_be_added_in_portfolio = False
        - Understand the user's query and call the extract_relevant_data_from_user_prompt function with the appropriate arguments like above.
        - Even though the user has asked for multiple tickers, you must strictly call the extract_relevant_data_from_user_prompt function only once with all the tickers in an array.
        """,
        functions=[extract_relevant_data_from_user_prompt],
    )
    stock_data_bot = ConversableAgent(
        name="stock_data_bot",
        system_message="""You are a Stock data gathering agent in this pipeline.
        
        Your specific role is to use the gather_stock_data function to get the relevant stock's data from the external APIs. Even though the user has asked for multiple tickers, you must strictly call the gather_stock_data function only once. The context_variables will contain all the necessary information to call the gather_stock_data function.        
        """,
        functions = [gather_stock_data]
    )
    cash_allocation_bot = ConversableAgent(
        name= "cash_allocation_bot",
        system_message="""You are a cash allocation agent in this pipeline.
        
        Your specific role is to use the the allocate_cash function to perform some mathematical calculations to calculate your stock portfolio returns. Even though the user has asked for multiple tickers, you must strictly call the allocate_cash function only once. The amount of dollars to be invested should be taken from the messages array correctly.
        """,
        functions= [allocate_cash]
    )
    insights_bot = ConversableAgent(
        name="insights_bot",
        system_message="""You are a insights agent in this pipeline.
        
        Your specific role is to use the generate_insights function to generate insights for the list of tickers in the context_variables.get('be_arguments')['ticker_symbols']. Generate 2 bull insights and 2 bear insights for each ticker. Even though the user has asked for multiple tickers, you must strictly call the generate_insights function only once. The context_variables will contain all the necessary information to call the generate_insights function.
        """,
        functions = [generate_insights]
    )

