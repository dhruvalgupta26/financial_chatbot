import json
import time
from bs4 import BeautifulSoup
import re
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import load_tools, AgentType, Tool, initialize_agent
import yfinance as yf

import warnings
import os
warnings.filterwarnings("ignore")

groq_api_key = "your api key"


llm=ChatGroq(
           model="Llama-3.1-70b-Versatile",
           groq_api_key=groq_api_key)


# Fetch stock data from Yahoo Finance
def get_stock_price(ticker,history=5):
    # time.sleep(4) #To avoid rate limit error
    if "." in ticker:
        ticker=ticker.split(".")[0]
    ticker=ticker+".NS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df=df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    df=df[-history:]
    # print(df.columns)
    
    return df.to_string()

# Script to scrap top5 googgle news for given company name
def google_query(search_term):
    # Ensure search_term is a string
    if isinstance(search_term, tuple):
        search_term = " ".join(search_term)
        print(type(search_term))
    if "news" not in search_term:
        search_term=search_term+" stock news"
    url=f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url=re.sub(r"\s","+",url)
    return url

def get_recent_stock_news(ticker):
    # time.sleep(4) #To avoid rate limit error
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    g_query=google_query(ticker)
    res=requests.get(g_query,headers=headers).text
    soup=BeautifulSoup(res,"html.parser")
    news=[]
    for n in soup.find_all("div","n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div","IJl0Z"):
        news.append(n.text)

    if len(news)>6:
        news=news[:4]
    else:
        news=news
    news_string=""
    for i,n in enumerate(news):
        news_string+=f"{i}. {n}\n"
    top5_news="Recent News:\n\n"+news_string
    
    return top5_news

# Fetch financial statements from Yahoo Finance
def get_financial_statements(ticker):
    # time.sleep(4) #To avoid rate limit error
    if "." in ticker:
        ticker=ticker.split(".")[0]
    else:
        ticker=ticker
    ticker=ticker+".NS"    
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>=3:
        balance_sheet=balance_sheet.iloc[:,:3]    # Remove 4th years data
    balance_sheet=balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

def get_stock_ticker(query):
    # Define the function for getting stock ticker
    function = {
        "name": "get_company_stock_ticker",
        "description": "Returns the company name and stock ticker symbol based on the query",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "ticker_symbol": {"type": "string"}
            },
            "required": ["company_name", "ticker_symbol"]
        }
    }

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="Given the user request, what is the company name according to this {query} ticker from yfinance? Return in JSON format Company_name and Ticker."
    )

    # Create the LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    # Perform the completion request
    response = chain.run({
        "query": query
    })

    # Print the raw response for inspection
    print("Raw Response:", response)

    # Attempt to extract the JSON part of the response
    try:
        # Find the start of the JSON part
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end != -1:
            # Extract the JSON substring
            json_str = response[start:end]
            # Load response as JSON
            response_json = json.loads(json_str)

            # Extract the company name and ticker symbol
            company_name = response_json.get('company_name', 'N/A')
            company_ticker = response_json.get('stock_ticker', 'N/A')

            # print(f"Company Name: {company_name}")
            # print(f"Ticker Symbol: {company_ticker}")

            # Return values
            return company_name
        else:
            print("No valid JSON found in response.")
            return None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None, None

def analyze_stock(query):
    # Retrieve the company name and ticker
    ticker = query
    company_name = get_stock_ticker(query)
    print({"Query": query, "Company_name": company_name, "Ticker": ticker})
    
    # Fetch stock data, financials, and news (implement these functions as needed)
    stock_data = get_stock_price(ticker, history=10)
    print(stock_data)
    stock_financials = get_financial_statements(ticker)
    print(stock_financials)
    stock_news = get_recent_stock_news(ticker)
    print(stock_news)
    # Format available information
    available_information = f"Stock Price: {stock_data}\n\nStock Financials: {stock_financials}\n\nStock News: {stock_news}"
    print(available_information)
    # Create the prompt for analysis
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give a detailed stock analysis using the available data and provide an investment recommendation. The user is fully aware of the investment risks. Do not include any warnings such as 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in your answer."),
            ("human", "{query} You have the following information available about {company_name}. Write (5-8) pointwise investment analysis to answer the user query. At the end, conclude with a proper explanation. Provide both positives and negatives:\n\n{available_information}"),
        ]
    )

    chain = prompt | llm
    # Perform the completion request for analysis
    response = chain.invoke({"query":query,"company_name":company_name,"available_information":available_information})
    
    # Extract and return the analysis
    analysis = response.content
    # print(analysis)
    return analysis
