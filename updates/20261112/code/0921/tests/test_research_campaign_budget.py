from scripts.run_research_campaign import admission, gate


def test_balance_guard_reserves_whole_jobs_before_calls():
    registration = {"full_campaign_model_token_cap": 32000000,
                    "balance_reserve_cny": 5, "conservative_cny_per_million_tokens": 8}
    jobs = [{"token_budget": 6000000}, {"token_budget": 2000000}]
    assert admission(jobs, {"model_tokens": 0}, registration, 68) == "ACCOUNT_BALANCE"
    assert admission(jobs, {"model_tokens": 0}, registration, 69) is None
    assert admission(jobs, {"model_tokens": 25000000}, registration, 100) == "TOKEN_CAP"


def test_feasibility_gate_never_filters_negative_quality_results():
    rows = [{"status": "COMPLETED_SEARCH", "candidate_budget": 12, "valid_candidates": 9,
             "best_train_gap": 99, "baseline_train_gap": 1}] * 2
    assert gate(rows) == "PASS"
    assert gate([rows[0], {"status": "INCOMPLETE"}]) == "PENDING"
