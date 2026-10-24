import json
import unittest
from types import SimpleNamespace as NS
from plan_execute import PlanAndExecute, execute_tool

class Message:
    def __init__(self, content=None, name=None, args=None):
        self.content = content
        self.tool_calls = [] if name is None else [NS(id='call_test', function=NS(name=name, arguments=json.dumps(args)))]
    def model_dump(self, **kwargs):
        d = {'role': 'assistant', 'content': self.content}
        if self.tool_calls:
            c = self.tool_calls[0]
            d['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': vars(c.function)}]
        return d

class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.chat = NS(completions=self)
    def create(self, **kwargs):
        x = next(self.outputs)
        m = x if isinstance(x, Message) else Message(json.dumps(x))
        return NS(choices=[NS(message=m, finish_reason='stop')])

def decision(status, steps=None, answer=''):
    return dict(status=status, steps=steps or [], answer=answer)

class Tests(unittest.TestCase):
    def test_full_flow(self):
        c = FakeClient([{'steps':['查订单','查物流']}, Message(name='get_order', args={'order_id':'A123'}), Message('SF001'), decision('continue',['查物流']), Message(name='get_shipment', args={'tracking_number':'SF001'}), Message('运输中'), decision('completed',answer='运输中')])
        r = PlanAndExecute(c,'fake',verbose=False).run('查A123')
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['history'][1]['tool_events'][0]['result']['location'],'上海转运中心')
    def test_blocked(self):
        c = FakeClient([{'steps':['查订单']}, Message(name='get_order',args={'order_id':'UNKNOWN'}),Message('订单不存在'),decision('blocked',answer='核对订单号')])
        r = PlanAndExecute(c,'fake',verbose=False).run('查UNKNOWN')
        self.assertEqual(r['status'],'blocked')
        self.assertIn('error',r['history'][0]['tool_events'][0]['result'])
    def test_replan_removes_step(self):
        c = FakeClient([{'steps':['查订单','查物流']},Message(name='get_order',args={'order_id':'A124'}),Message('未发货'),decision('completed',answer='未发货')])
        r=PlanAndExecute(c,'fake',verbose=False).run('查A124')
        self.assertEqual(len(r['history']),1)
    def test_tool_errors(self):
        for name,args in [('other','{}'),('get_order','bad'),('get_order','[]'),('get_order','{"order_id":1}'),('get_order','{"order_id":""}'),('get_order','{"wrong":"A123"}')]:
            self.assertIn('error',execute_tool(name,args))
    def test_json_repair(self):
        c=FakeClient([Message('bad'),{'steps':['查订单']}])
        self.assertEqual(PlanAndExecute(c,'fake',verbose=False).planner('查A123'),['查订单'])
    def test_invalid_json_limit(self):
        c=FakeClient([Message('bad'),{'steps':[]}])
        with self.assertRaises(ValueError):
            PlanAndExecute(c,'fake',verbose=False).planner('查A123')
    def test_limits(self):
        c=FakeClient([{'steps':['查订单']},Message(name='get_order',args={'order_id':'A123'}),decision('continue',['查物流'])])
        r=PlanAndExecute(c,'fake',max_steps=1,max_tool_rounds=1,verbose=False).run('查A123')
        self.assertEqual(r['status'],'limit_reached')
        self.assertEqual(r['history'][0]['status'],'limit_reached')

if __name__ == '__main__':
    unittest.main()
