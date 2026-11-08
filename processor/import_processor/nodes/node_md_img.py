import base64
import re
import sys
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

import common.config.lm_config
from common.config import lm_config
from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task


def step_1_validate_and_get_data(state):
    """
       提取和校验内容,并且返回图片的地址
       :param state:
       :return: 返回md内容,md地址,images地址
       """
    md_path = state.get("md_path")
    if not md_path:
        logger.error("md_path变量为空!")
        raise ValueError("md_path变量为空!")
    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        logger.error(f"{md_path_obj}不存在!")
        raise FileNotFoundError(f"{md_path_obj}不存在!")

    md_content = md_path_obj.read_text(encoding="utf-8")
    state['md_content'] = md_content

    md_images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, md_images_dir_obj


def step_2_scan_images(md_content:str, md_images_dir_obj:Path)->list[tuple[str,str,tuple[str,str]]]:
    image_info_list = [];
    for image_file_obj in md_images_dir_obj.iterdir():
        image_name = image_file_obj.name
        image_path = str(image_file_obj)

        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match = pattern.search(md_content)
        if not search_match:
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue
        start = search_match.start()
        end = search_match.end()

        pre_context = md_content[max(0, start - 100):start]
        post_context = md_content[end:min(end + 100, len(md_content))]
        logger.debug(
            f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_context} , 下文:{post_context}")
        image_info_list.append(
            (image_name, image_path, (pre_context, post_context))
        )
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list


def step_3_image_summary(image_info_list, root_folder:str):
    summary_image_dict: dict[str, str] = {}
    # 1.准备模型对象
    vl_model = get_llm_client(model=lm_config.vl_model)
    # 2.循环image_info_list数据,获取每一张图片的信息
    for image_name, image_path, image_content in image_info_list:
        # 3.循环里,将图片的信息拼接成提示词
        # 视觉模型->初步知道-> 1. 图片 2.文本
        image_prompt_text = load_prompt("image_summary", root_folder=root_folder, image_content=image_content)
        image_path_obj: Path = Path(image_path)
        image_base64_str: str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"}
                },  # 图片
                {
                    "type": "text", "text": image_prompt_text
                }  # 文本
            ]
        )
        # 4.调用视觉模型获取结果
        # 调用链 chains
        chains = vl_model | StrOutputParser()
        # 添加范围内限制
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        # 5.结果拼接到字典数据中
        summary_image_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
    # 6.循环外返回字典数据
    return summary_image_dict


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    add_running_task(state['task_id'], 'node_md_img')
    md_content, md_path_obj, md_images_dir_obj = step_1_validate_and_get_data(state)
    
    if (not md_images_dir_obj.is_dir()) or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state

    image_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)
    summary_image_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)
    image_url_dict: dict[str, str] = step_4_upload_images_get_url(image_info_list, md_path_obj.stem)
    md_content_new: str = step_5_md_content_image_replace(md_content, summary_image_dict, image_url_dict)
    md_path_obj_new: Path = step_6_backup_new_md_content(md_content_new, md_path_obj)
    state['md_content'] = md_content_new
    state['md_path'] = str(md_path_obj_new)
    add_done_task(state['task_id'], 'node_md_img')
    return state