# third_party_vuln







## Dataset

Currently we have crafted two datasets

**Coarse-grained features**: textual `comment` only

- [SUSE](dataset/suse.csv)
- [KDE](dataset/KDE.csv)
- [Mozilla](dataset/mozilla.csv)
- [Gentoo](dataset/gentoo.csv)
- [VSCode](dataset/VSCode.csv)

**Fine-grained features**: subcategory, comment->(telementry,crash information, backtrace)

- [Thunderbird](dataset/thunderbird_featured.csv)
- [Firefox](dataset/firefox_featured.csv)
- [KDE](dataset/KDE_featured.csv)

## RAG

`rag.py` has implemented two functionalities:
- initialize() to create embedding in qdrant database using LLAMA or OPENAI (defined at `embedding.py')
- query() to find the most similar comments as the given one from the qdrant database 