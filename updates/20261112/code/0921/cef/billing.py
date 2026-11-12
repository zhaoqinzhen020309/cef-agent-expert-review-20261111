"""Provider usage and account-balance evidence. Never estimates a charge or limits search."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from .errors import exception_chain


def summarize_usage(charges):
    usage=[row for row in charges if row.get('resource')=='llm_tokens']
    known=[row for row in usage if row.get('usage_known') is True]
    unknown=sum(row.get('usage_known') is not True for row in usage)
    total=sum(row['amount'] for row in known)
    return dict(requests=sum(row['amount'] for row in charges if row.get('resource')=='llm_requests'),
                known_input_tokens=sum(row.get('input_tokens',0) for row in known),
                known_cached_input_tokens=sum(row.get('cached_input_tokens',0) for row in known),
                known_output_tokens=sum(row.get('output_tokens',0) for row in known),
                known_total_tokens=total,total_tokens=None if unknown else total,
                unknown_usage_requests=unknown,provider_reported_cost=None,
                cost_note='Generation API returns usage, not an authoritative per-request debit amount.')


def balance_changes(before,after):
    if 'balance_infos' not in before or 'balance_infos' not in after: return None
    left={row['currency']:row for row in before['balance_infos']}
    right={row['currency']:row for row in after['balance_infos']}
    return [dict(currency=currency,balance_decrease=format(Decimal(left[currency]['total_balance'])-Decimal(right[currency]['total_balance']),'f'),
                 scope='account-wide balance change, not per-request invoice')
            for currency in sorted(left.keys()&right.keys())]


def balance_snapshot(config):
    key=os.environ.get(config['api_key_env'])
    async def read():
        async with asyncio.timeout(15):
            async with AsyncOpenAI(api_key=key,base_url=config['endpoint'],max_retries=0,timeout=15,
                http_client=DefaultAsyncHttpxClient(trust_env=config['route']=='environment_proxy')) as client:
                base=config['endpoint'].rstrip('/')
                if base.endswith('/v1'): base=base[:-3]
                response=await client.get(base+'/user/balance',cast_to=httpx.Response)
                return response.json()
    observed=datetime.now(timezone.utc).isoformat()
    try:
        if not key: raise ValueError('API credential unavailable for balance lookup')
        return dict(observed_at=observed,**asyncio.run(read()))
    except Exception as error:
        # Accounting failure never becomes a search stopping condition.
        return dict(observed_at=observed,error_chain=exception_chain(error,key))


def write_billing(folder,before,after):
    folder=Path(folder); charges=[]; raw=[]
    for stage in ('preflight','search'):
        path=folder/stage/'facts.jsonl'
        if not path.is_file(): continue
        records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        stage_charges=[row['data'] for row in records if row['kind']=='charge']
        requests=sum(row['kind']=='provider_request' for row in records)
        settled=sum(row.get('resource')=='llm_tokens' for row in stage_charges)
        for _ in range(max(0,requests-settled)):
            stage_charges.append(dict(resource='llm_tokens',amount=0,usage_known=False,unresolved_request=True))
        charges.extend(stage_charges)
        raw.extend(dict(stage=stage,**row['data']) for row in records if row['kind']=='provider_response')
    result=dict(usage=summarize_usage(charges),account_balance_changes=balance_changes(before,after),
                authoritative_per_request_debit=None,
                attribution_note='Other clients, recharge, grant expiry and settlement timing can affect account-wide changes. Provider console bills govern attribution.')
    for name,value in [('balance_before.json',before),('balance_after.json',after),
                       ('billing_summary.json',result),('provider_usage.json',[dict(stage=row['stage'],usage=row.get('usage'),error_chain=row.get('error_chain')) for row in raw])]:
        with (folder/name).open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    return result
