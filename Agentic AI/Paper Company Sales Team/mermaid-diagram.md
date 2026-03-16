flowchart LR
    Customer([Customer Request]) --> Orchestrator

    subgraph Orchestrator["🧠 Orchestrator Agent\nParses request · routes · returns reply"]
    end

    Orchestrator -->|stock query| INV
    Orchestrator -->|quote request| QUOT
    Orchestrator -->|place order| ORD
    Orchestrator -->|internal result| COMM

    subgraph INV["📦 Inventory Agent\nChecks stock · places reorders"]
        T1["check_inventory\nStock level + reorder flag\n→ get_stock_level()"]
        T2["get_full_inventory\nAll item quantities\n→ get_all_inventory()"]
        T3["reorder_stock\nSupplier order at order date\n→ create_transaction()"]
    end

    subgraph QUOT["💰 Quoting Agent\nQuotes prices · applies discounts"]
        T4["get_quote_history\nSearch past quotes by keyword\n→ search_quote_history()"]
        T5["calculate_quote\n5/10/15% at 100/500/1000 units\n→ get_stock_level()"]
        T6["check_inventory\nVerify availability before quoting\n→ get_stock_level()"]
    end

    subgraph ORD["🛒 Ordering Agent\nFulfills sales · restocks after sale"]
        T7["fulfill_order\nValidates stock · records sale\n→ create_transaction()"]
        T8["get_delivery_estimate\nLead time from order quantity\n→ get_supplier_delivery_date()"]
        T9["get_cash\nCash balance before committing\n→ get_cash_balance()"]
        T10["get_financial_report\nFull report after sale\n→ generate_financial_report()"]
        T11["reorder_stock\nRestocks if stock drops post-sale\n→ create_transaction()"]
    end

    subgraph COMM["✉️ Communications Agent\nNo tools · rewrites internal output only"]
        T12["Rewrite rule: strip internal errors\nUnknown item · transaction IDs · system refs\n→ plain business language"]
        T13["Rewrite rule: enrich successes\nAdd pricing · discounts · delivery date\n→ customer-facing response"]
    end

    COMM -->|clean response| Customer

    DB[("SQLite Database\ntransactions · inventory · quotes · quote_requests")]

    T1 & T2 & T3 --> DB
    T4 & T5 & T6 --> DB
    T7 & T8 & T9 & T10 & T11 --> DB