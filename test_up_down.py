
def getStockFromQuandl():
    import quandl
    df = quandl.get("WIKI/AAPL", start_date="2000-01-01", end_date="2017-07-15")

    df['Up or Down'] = df['Adj. Volume'].pct_change()

    return df


sth = getStockFromQuandl()
sth.fillna(0, inplace=True)
print(sth)
sth2 = [1 if v > 0 else -1 for v in sth['Up or Down']]
print(sth2)

