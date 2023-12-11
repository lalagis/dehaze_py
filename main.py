import cv2
import time
import base64
import flet as ft
import numpy as np
from utils.raster import Raster, save_geotiff
from utils.algorithm import (
    darkChannelPriorDehaze,
    homomorphicFilterDeHaze,
    zmIceColor,
    equalizeHistDehaze,
    retinexDehaze,
    nafnetPredict
)


def main(page: ft.Page):
    # 基础设置
    page.title = "遥感影像去雾的几种算法实现"
    page.theme = ft.Theme(color_scheme_seed="green")
    page.update()

    # 原始影像实例
    raster: Raster | None = None
    # 当前渲染图像
    current_image: np.ndarray | None = None

    # 将影像以base64编码渲染
    def render_image(img: np.ndarray):
        global current_image
        current_image = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base64_str = cv2.imencode(".png", img)[1].tostring()
        base64_str = str(base64.b64encode(base64_str))[2:-1]

        container.content = ft.Image(
            src_base64=base64_str, width=500, height=500, fit=ft.ImageFit.CONTAIN
        )
        container.update()

    # 重置原始影像
    def reset_origin_image():
        global raster
        if raster is not None:
            img = raster.getArray()
            render_image(img)
            rail.selected_index = None
            rail.leading = choose_image_button
            rail.update()

    # 当选择了原始影像后
    def on_pick_image(e: ft.FilePickerResultEvent):
        global raster
        if e.files:
            raster = Raster(e.files[0].path)
            reset_origin_image()

    # 文件选择器
    image_picker = ft.FilePicker(on_result=on_pick_image)

    # 当点击下载按钮后
    def on_save(e: ft.FilePickerResultEvent):
        global current_image
        global raster
        if e.path:
            save_geotiff(
                image=current_image,
                save_path=e.path,
                proj=raster.proj,
                geotf=raster.geot,
            )
            page.snack_bar = ft.SnackBar(ft.Text("保存成功"))
            page.snack_bar.open = True
            page.update()

    # 文件保存器
    save_picker = ft.FilePicker(on_result=on_save)

    # 当选择了不同算法
    def on_change_algorithm(e):
        global raster
        if raster is not None:
            rail.leading = show_origin_image_button
            rail.update()
            img = raster.getArray()
            start = time.time()
            if rail.selected_index == 0:
                img = equalizeHistDehaze(img)
            if rail.selected_index == 1:
                img = retinexDehaze(img)
            if rail.selected_index == 2:
                img = zmIceColor(img)
            if rail.selected_index == 3:
                img = homomorphicFilterDeHaze(img)
            if rail.selected_index == 4:
                img = darkChannelPriorDehaze(img)
            if rail.selected_index == 5:
                img = nafnetPredict(img)
            end = time.time()
            render_image(img)
            show_time_message(end - start)
        else:
            print("请先选择原始影像")

    # 两种不同的侧边栏头部按钮
    choose_image_button = ft.ElevatedButton(
        "选择影像",
        icon=ft.icons.IMAGE_ROUNDED,
        on_click=lambda _: image_picker.pick_files(),
    )
    show_origin_image_button = ft.ElevatedButton(
        "原始影像", icon=ft.icons.IMAGE_ROUNDED, on_click=lambda _: reset_origin_image()
    )

    # 运行时间消息弹出
    def show_time_message(time):
        page.snack_bar = ft.SnackBar(ft.Text(f"运行时间：{time * 1000} ms"))
        page.snack_bar.open = True
        page.update()

    # 侧边栏
    rail = ft.NavigationRail(
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=400,
        group_alignment=-0.9,
        # 选择影像按钮
        leading=choose_image_button,
        # 算法选择
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.icons.ANALYTICS_OUTLINED,
                selected_icon_content=ft.Icon(
                    name=ft.icons.ANALYTICS, color=ft.colors.GREEN_800
                ),
                label="直方图均衡化",
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.REMOVE_RED_EYE_OUTLINED,
                selected_icon_content=ft.Icon(
                    name=ft.icons.REMOVE_RED_EYE, color=ft.colors.GREEN_800
                ),
                label="Retinex算法",
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.COLOR_LENS_OUTLINED,
                selected_icon=ft.Icon(
                    name=ft.icons.COLOR_LENS, color=ft.colors.GREEN_800
                ),
                label="ACE算法",
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.WAVES_OUTLINED,
                selected_icon_content=ft.Icon(
                    name=ft.icons.WAVES, color=ft.colors.GREEN_800
                ),
                label="同态滤波",
            ),
            # ft.Divider(height=1),
            ft.NavigationRailDestination(
                icon=ft.icons.CLOUD_OUTLINED,
                selected_icon_content=ft.Icon(
                    ft.icons.FOGGY, color=ft.colors.GREEN_800
                ),
                label="暗通道先验",
            ),
            # ft.Divider(height=1),
            ft.NavigationRailDestination(
                icon=ft.icons.ACCOUNT_TREE_OUTLINED,
                selected_icon_content=ft.Icon(
                    ft.icons.ACCOUNT_TREE_SHARP, color=ft.colors.GREEN_800
                ),
                label="NAFNET去噪模型",
            ),
        ],
        on_change=on_change_algorithm,
    )

    # 分割线
    devider = ft.VerticalDivider(width=1)

    # 容器
    container = ft.Container(
        content=ft.Text("暂未选择原始影像"),
        alignment=ft.alignment.center,
        width=600,
        height=600,
    )

    # 横向布局
    row = ft.Row([rail, devider, image_picker, container, save_picker], expand=True)

    # 下载按钮
    page.floating_action_button = ft.FloatingActionButton(
        icon=ft.icons.SAVE,
        bgcolor=ft.colors.GREEN_400,
        tooltip="保存当前影像",
        on_click=lambda _: save_picker.save_file(),
    )

    # 页面
    page.add(row)


# 应用
ft.app(target=main)
