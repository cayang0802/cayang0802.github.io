---
title: 優化 RAG 檢索的數個方法
description: RAG (Retrieval Augmented Generation) 透過外部來源取得資訊，並將資訊提供給 LLM 產生回答，成功地解決了 LLM 回答受限於其訓練資料的問題。
slug: optimize-rag-retrieval-methods
date: 2026-05-12 00:00:00+0000
image:
categories:
    - AI
tags:
    - AI
    - RAG
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

RAG (Retrieval Augmented Generation) 透過外部來源取得資訊，並將資訊提供給 LLM 產生回答，成功地解決了 LLM 回答受限於其訓練資料的問題。但是我們希望檢索到的答案是我們想要的，否則只是一坨垃圾話。

因此這邊整理了一些或許有用的方法，當檢索成效不如預期時，不妨參考看看嘿

我分成三個階段做說明，分別是**前檢索階段、檢索階段、後檢索階段**

**前檢索階段**：資料前處理，優化資料索引與使用者查詢

**檢索階段**：優化向量搜尋

**後檢索階段**：過濾檢索到的資訊

## **前檢索階段**

此階段重點是如何將資料建立索引 (index)，以及檢索向量前可以優化的方法

**建立索引**

1. **滑動視窗 (Sliding Window)**：相鄰的文字分段 (chunk) 邊緣之間產生重疊部分，避免重要資訊發生在邊緣時被切開來，然後遺失
2. **注意切割邊緣**：切割時請避免斷在句子一半。以段落結尾為斷點會是相對好的方法。像是 LangChain 的 *RecursiveCharacterTextSplitter* 通常比 *TokenTextSplitter* 來得好
3. **添加 Metadata**： 附上日期、URL、章節、頁面等資訊，有助於後續檢索能過濾結果
4. **優化索引結構**：使用不同索引技巧，像是使用多種 chunk size 來切分。像是 PDF 內的資料有文字圖片穿插，怎麼切特別重要
5. **資料清洗**：去除垃圾話、驗證事實正確性、更新過時資訊等
6. **由小到大檢索 (Small-to-Big Retrieval)**：用一小段中最重要的部分轉化為嵌入作為索引，並將周圍的上下文保留在 metadata 中。生成回應時再提供給 LLM。解決了小區塊不含雜訊但資訊量不足，大區塊資訊充足但雜訊太多的問題。LangChain 可以參考 *ParentDocumentRetriever*

**檢索向量前**

1. **查詢路由 (query routing)**：當資料爆多時，可以考慮將資料分類。例如資料有文章、程式碼兩種，可以將資料庫分兩個；在使用者問程式碼相關的問題時，將後續檢索範圍導到程式碼資料庫中。增加檢索效率的同時也減少檢索錯誤資料的機會
2. **查詢改寫 (query rewriting)**：將使用者的查詢換句話說，但保留原本意思，增加與向量資料庫匹配的機會。例如將較少見的詞彙換成較常見的，擴大搜尋範圍；對於較長的 query 也可以考慮拆成多個 sub-query，分開檢索
3. **HyDE (Hypothetical Document Embeddings)**：透過 LLM 先對 query 生成一個假設性回應，然後用原始 query 與上述假設回應進行檢索
4. **查詢擴展 (query expansion)**：在使用者的 query 加入額外字詞讓查詢更全面。例如有人搜尋『沙發』，可以加上同義詞像是『長椅』或『Couch』
5. **關鍵字過濾**：先以 LLM 偵測查詢中如果出現特殊關鍵字例如人名地名，可以利用關鍵字過濾向量搜尋的範圍

## 檢索階段

檢索主要有兩種：基於語意的搜索 and 基於關鍵字的搜索

語意搜索其實就是比對 query vector 與 chunk vector 的向量相似度，相似度越高代表語意越接近

關鍵字搜索會利用出現文字與出現頻率去計算分數，方法像是 TF-IDF, BM25

1. **微調 Embedding 模型**：向量資料庫是從 embedding 來的，如果應用場合含有大量罕見詞彙，可以考慮微調模型
2. **利用 LLM 模型引導**：如果不想微調模型，也可以考慮用 Instructor Embedding Model (例如這個 [hkunlp/instructor-large · Hugging Face](https://huggingface.co/hkunlp/instructor-large)) 告訴目前任務背景，在 embedding 時去調整 query 到符合資料庫的資料
3. **善用 metadata 過濾與搜尋**：如果資料庫有附帶 metadata，可以用它幫忙過濾
4. **混合搜尋 (hybrid search)**：語意搜索+關鍵字搜索。兩種搜索各自有優缺點，因此利用兩種搜尋得到各自分數，然後利用 RRF (Reciprocal rank fusion) 根據權重算出最後的排名。權重可以根據領域做調整

## 後檢索階段

檢索出來的資料會餵給 LLM 協助生成答案。為了避免受到 LLM 的 context window size 等限制，這部分也是可以考慮優化的一個階段

1. **提示詞壓縮**：如果訊息太長，在保留核心訊息的前提下，將廢話去除。著名技術有 LLMLingua
2. **重新排序 (Re-ranking)**：將檢索後的資料排名重新排序。利用『使用者輸入』與『每一個檢索片段』的匹配分數去排序。匹配分數可以由 Bi-Encoder (兩個片段的向量相似度) 得到，或是利用 Cross-Encoder (Reranker) 得到。後者效果較好但速度較慢。此外HuggingFace 有提供許多 Reranker。要注意有些 Reranker 不支援中文。針對 Reranker 的研究可以看這一篇 [使用繁體中文評測各家 Reranker 模型的重排能力 &#8211; ihower { blogging }](https://ihower.tw/blog/12227-reranker)

-------------------------

另外現在也有基於圖的方法 GraphRAG，又名知識圖譜，著名工具有像是 Neo4j

不同的應用場景適合不同方法，沒有絕對好的方法，只有多實驗並評估才是真理
