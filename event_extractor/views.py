import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from django.http import HttpResponse, JsonResponse
import json
from event_extractor.train.train import TrainProcess
from event_extractor.predict.event_pipeline import EventExtractor

lang_model = {}


def index(request):
    return HttpResponse("index page")


def create_train(request):
    if request.method != "POST":
        return JsonResponse({"success": "false", "message": "request method must via POST！"})
    req_body = request.body.decode("utf-8")
    receive_json = json.loads(req_body)
    params = receive_json["params"]
    algo_name = params["algo_type"]
    result_dict = {}
    result_dict["success"] = "True"
    result_dict["message"] = "Successfully!"
    result_dict["result_list"] = {}
    if algo_name == "entity":
        # try:
        entity_instance = TrainProcess(params)
        entity_instance.load_data()
        entity_instance.train()
    # except:
    #     result_dict["message"] = "failed!"
    ret_obj = JsonResponse(result_dict, safe=False)
    return ret_obj


def event_extract(request):
    if request.method != "POST":
        return JsonResponse({"success": "false", "message": "request method must via POST！"})
    req_body = request.body.decode("utf-8")
    receive_json = json.loads(req_body)
    lang = receive_json["language"]
    model_id = receive_json["model_id"]
    if lang not in lang_model:
        lang_model[lang] = EventExtractor(lang=lang, model_id=model_id)
    result_dict = {}
    result_dict["success"] = "True"
    result_dict["message"] = "Successfully!"
    result_dict["result_list"] = lang_model[lang].get_event_result(receive_json)

    ret_obj = JsonResponse(result_dict, safe=False)
    return ret_obj
