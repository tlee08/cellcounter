from cellcounter.utils.template_utils import import_static_templates_script


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    import_static_templates_script(
        description="Make Cellcounter Pipeline Script",
        templates_ls=["run_pipeline.py", "view_img.py"],
        pkg_name="cellcounter",
        pkg_subdir="templates",
        root_dir=".",
        overwrite=False,
        dialogue=True,
    )
    # # Copying default configs file to the project folder
    # write_json(os.path.join(root_dir, "default_configs.json"), ConfigParamsModel().model_dump())


if __name__ == "__main__":
    main()
