import os
import sys
import cairosvg



def convert_svgs(folder):
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".svg"):
            old_svg_path = os.path.join(folder, file_name)

            # Remove '-solid' from file name if present
            new_svg_name = file_name.replace("-solid.svg", ".svg")
            new_svg_path = os.path.join(folder, new_svg_name)
            if new_svg_path != old_svg_path:
                os.rename(old_svg_path, new_svg_path)

            # Convert SVG to PNG (scaled to add some padding)
            png_name = new_svg_name.replace(".svg", ".png")
            png_path = os.path.join(folder, png_name)
            cairosvg.svg2png(url=new_svg_path, write_to=png_path, scale=2)


if __name__ == "__main__":
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, "masks")
    if len(folder_path) < 2:
        print("Converting folder")
        sys.exit(1)
    convert_svgs(folder_path)
