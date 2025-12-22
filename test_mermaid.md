```mermaid
graph LR
    %% 스타일 정의
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    style B fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
    style C fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000

    %% 1단계: Input
    subgraph Input_Layer [<b>STEP 1: Input (Unstructured Data)</b>]
        direction TB
        A[<b>Google Maps Review Data</b><br/>(Raw Text)]
        A1[Target: 2,500 Cafes in Seoul]
        A2[Preprocessing: Sentence Splitting]
        A --> A1
        A --> A2
    end

    %% 2단계: System
    subgraph System_Layer [<b>STEP 2: Placeness Quantification System</b>]
        direction TB
        B[<b>Hybrid NLP Model</b>]
        
        B1(<b>1. Semantic Filtering</b><br/>Model: Sentence-BERT<br/>Function: Cosine Similarity ≥ 0.4)
        
        B2(<b>2. Sentiment Scoring</b><br/>Model: KoBERT<br/>Output: Sentiment Score 0~1)
        
        B3(<b>3. Weighted Aggregation</b><br/>Calculation: Factor Score fsi × Weight wi<br/>Result: Comprehensive Score μ)

        B --> B1
        B1 --> B2
        B2 --> B3
    end

    %% 3단계: Output
    subgraph Output_Layer [<b>STEP 3: Output (Structured Metrics)</b>]
        direction TB
        C[<b>Quantitative Place Metrics</b>]
        C1[<b>Comprehensive Score (μ)</b><br/>Total Placeness Value]
        C2[<b>Deviant Features (df+, df-)</b><br/>Strong/Weak Characteristics]
        C3[<b>Visualization</b><br/>Radar Chart & Heatmap]
        
        C --> C1
        C --> C2
        C --> C3
    end

    %% 연결선
    Input_Layer ==> System_Layer
    System_Layer ==> Output_Layer
```

