try:
    import pandas as pd
    import textblob
    import matplotlib
    import seaborn
    import streamlit
    import nltk

    print("Installed")

except ImportError as e:
    print("Missing:", e.name)
