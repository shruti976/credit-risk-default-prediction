"""Download the UCI 'Default of Credit Card Clients' dataset into ./data/."""
import os, urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
DEST = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DEST, exist_ok=True)
out = os.path.join(DEST, "credit.xls")
print("Downloading UCI default-of-credit-card-clients ...")
urllib.request.urlretrieve(URL, out)
print("Saved to", out)
