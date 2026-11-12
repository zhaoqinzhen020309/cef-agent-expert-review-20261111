import asyncio,json
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import pytest
from openai import AsyncOpenAI
from cef.model import Model


def receive(frames,drop=False,state=None):
    body=''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+'data: [DONE]\n\n'
    calls=[]
    def handler(request):
        calls.append(json.loads(request.content))
        if drop:
            class BrokenBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield body.encode()
                    raise httpx.RemoteProtocolError('peer closed connection')
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=BrokenBody())
        return httpx.Response(200,headers={'content-type':'text/event-stream','x-request-id':'req-stream'},content=body)
    factory=lambda **kwargs:AsyncOpenAI(**dict(kwargs,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    model=Model.__new__(Model)
    model.key='test-only';model.endpoint='https://example.test';model.config={'route':'direct'}
    if state is None:state=dict(message=None,usage=None,finish=None,request_id=None,provider_response_id=None)
    with patch('cef.model.AsyncOpenAI',factory):
        asyncio.run(model._receive_native({'model':'test','messages':[],'stream':True,'stream_options':{'include_usage':True}},2,state))
    return state,calls


def chunk(delta=None,finish=None,usage=None):
    return dict(id='completion',object='chat.completion.chunk',created=0,model='test',
                choices=[] if usage else [dict(index=0,delta=delta or {},finish_reason=finish)],usage=usage)


def test_stream_retains_reasoning_and_interleaved_tool_calls():
    state,calls=receive([
        chunk({'role':'assistant','reasoning_content':'think '}),
        chunk({'reasoning_content':'more','tool_calls':[{'index':0,'id':'a','type':'function','function':{'name':'check','arguments':'{"x":'}}]}),
        chunk({'tool_calls':[{'index':1,'id':'b','type':'function','function':{'name':'read','arguments':'{"p":"'}}]}),
        chunk({'tool_calls':[{'index':0,'function':{'arguments':'1}'}},{'index':1,'function':{'arguments':'文件"}'}}]}),
        chunk(finish='tool_calls'),chunk(usage={'prompt_tokens':10,'completion_tokens':20,'total_tokens':30})])
    assert state['message']['reasoning_content']=='think more'
    assert [t['id'] for t in state['message']['tool_calls']]==['a','b']
    assert [t['function']['arguments'] for t in state['message']['tool_calls']]==['{"x":1}','{"p":"文件"}']
    assert state['usage']['total_tokens']==30 and state['finish']=='tool_calls'
    assert state['request_id']=='req-stream' and len(calls)==1


@pytest.mark.parametrize('frames',[
    [chunk({'role':'assistant','content':'half'})],
    [chunk({'role':'assistant','content':'half'},finish='length')],
    [chunk({'role':'assistant','content':'complete'},finish='stop')],
])
def test_incomplete_stream_never_publishes_usable_message(frames):
    with pytest.raises(ValueError,match='incomplete|usage'):
        receive(frames)


def test_actual_sdk_disconnect_retains_observation_but_no_executable_message():
    state=dict(message=None,usage=None,finish=None,request_id=None,provider_response_id=None)
    with pytest.raises(httpx.RemoteProtocolError):
        receive([chunk({'role':'assistant','tool_calls':[{'index':0,'id':'half','type':'function',
                  'function':{'name':'write','arguments':'{"source":"'}}]})],drop=True,state=state)
    assert state['message'] is None and state['usage'] is None
    assert state['stream_observation']['chunks']==1
