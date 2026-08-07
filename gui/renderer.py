CELL_SIZE = 30

CELL_COLORS = {
    "#": "#2f3542",
    ".": "#f1f2f6",
    "S": "#70a1ff",
    "P": "#ff6b81",
    "T": "#a55eea",
    "K": "#feca57",
    "D": "#8d6e63",
    "G": "#2ed573"
}

FLOOR_COLOR = "#f1f2f6"
OPEN_DOOR_COLOR = "#c8e6c9"
GRID_COLOR = "#ced6e0"
AGENT_COLOR = "#1e90ff"


def draw_maze(
    canvas,
    maze,
    agent_position,
    has_key
):
    canvas.delete("all")

    for row_index, maze_row in enumerate(maze):
        for column_index, cell in enumerate(
            maze_row
        ):
            x1 = column_index * CELL_SIZE
            y1 = row_index * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            cell_color = CELL_COLORS.get(
                cell,
                FLOOR_COLOR
            )

            cell_label = ""

            if cell in [
                "S",
                "P",
                "T",
                "K",
                "D",
                "G"
            ]:
                cell_label = cell

            if cell == "K" and has_key:
                cell_color = FLOOR_COLOR
                cell_label = ""

            if cell == "D" and has_key:
                cell_color = OPEN_DOOR_COLOR

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=cell_color,
                outline=GRID_COLOR
            )

            if cell_label:
                canvas.create_text(
                    x1 + CELL_SIZE / 2,
                    y1 + CELL_SIZE / 2,
                    text=cell_label,
                    font=("Arial", 10, "bold")
                )

    agent_row = agent_position[0]
    agent_column = agent_position[1]

    agent_x1 = (
        agent_column * CELL_SIZE + 5
    )

    agent_y1 = (
        agent_row * CELL_SIZE + 5
    )

    agent_x2 = (
        agent_x1 + CELL_SIZE - 10
    )

    agent_y2 = (
        agent_y1 + CELL_SIZE - 10
    )

    canvas.create_oval(
        agent_x1,
        agent_y1,
        agent_x2,
        agent_y2,
        fill=AGENT_COLOR,
        outline="white",
        width=2
    )