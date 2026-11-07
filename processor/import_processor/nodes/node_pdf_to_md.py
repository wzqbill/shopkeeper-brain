import os
import shutil
import sys
import time
from pathlib import Path

import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.error(f"pdf_path的值为空,无法读取文件,直接抛出异常!")
        raise ValueError("pdf_path的参数值为空,无法读取文件!")
    if not local_dir:
        logger.warning(f"没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.is_file():
        logger.error(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
        raise ValueError(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir:{local_dir_obj}不存在，或者不是文件夹,我们主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        "model_version": "vlm"
    }
    http_response = requests.post(url, headers=header, json=data)
    http_response_status_code = http_response.status_code
    if http_response_status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{http_response_status_code},请检查minerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{http_response_status_code},请检查minerU配置!")
    response_dict = http_response.json()
    response_dict_code = response_dict.get("code")
    response_dict_msg = response_dict.get("msg")
    if response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
        raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")

    batch_id = response_dict.get("data").get("batch_id")
    file_upload_urls = response_dict.get("data").get("file_urls")

    if not file_upload_urls:
        logger.error(f"没有申请到上传地址")
        raise RuntimeError(
            f"没有申请到上传地址")
    if not batch_id:
        logger.error(f"没有批量操作标识")
        raise RuntimeError(
            f"没有批量操作标识")

    file_upload_url = file_upload_urls[0]
    logger.info(f">>>申请地址成功{file_upload_url}>>>")

    pdf_file_data = pdf_path_obj.read_bytes()
    with requests.session() as session:
        session.trust_env = False
        upload_response = session.put(file_upload_url, data=pdf_file_data)
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    logger.info(f">>>向{file_upload_url}上传{pdf_path_obj}成功>>>")

    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout = 600
    interval_time = 3
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")

        try:
            poll_response = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        http_poll_status_code = poll_response.status_code
        if http_poll_status_code != 200:
            if 500 <= http_poll_status_code < 600:
                # 如果是服务器报错  给机会继续重试
                logger.warning(f"可有修复的网络异常,状态码为:{http_poll_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")

        poll_response_dict = poll_response.json()
        if poll_response_dict['code'] != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")

        extract_result = poll_response_dict['data']['extract_result'][0]
        extract_result_state = extract_result['state']

        if extract_result_state == 'done':
            extract_result_url = extract_result['full_zip_url']
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
            # 5. 返回压缩地址即可
            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == 'failed':
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    response = requests.get(zip_url, timeout=60)
    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")
    md_zip_path_obj = local_dir_obj / f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(response.content)

    extract_path_obj = local_dir_obj / stem
    if extract_path_obj.is_dir():
        shutil.rmtree(extract_path_obj)

    extract_path_obj.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(md_zip_path_obj, extract_path_obj)

    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list:
        logger.error(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
        raise RuntimeError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")


    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            logger.info(f"文件{md_file_obj}解压成功")
            return md_file_obj


    target_md_obj = None
    for md_file_obj in md_file_list:
        if md_file_obj.name.lower() == "full.md":
            target_md_obj = md_file_obj
            logger.info(f"文件解压成功,但是名字是full.md,后续需要重命名")
            break

    if not target_md_obj:
        target_md_obj = md_file_list[0]

    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{md_file_obj}解压成功")
    return final_md_path_obj


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    add_running_task(state.get("task_id"), "node_pdf_to_md")
    pdf_path_obj, local_dir_obj = step_1_validate_and_get_data(state)
    zip_url = step_2_upload_and_poll(pdf_path_obj)
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    state['md_path'] = md_path
    add_done_task(state['task_id'], 'node_pdf_to_md')
    add_done_task(state.get("task_id"), "node_pdf_to_md")

    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "万用表RS-12的使用.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")