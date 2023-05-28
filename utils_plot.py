def plot_agent(annotation, ax):
    width = annotation['size'][1]
    height = annotation['size'][0]
    translation = annotation['translation']
    xy = (translation[0] - width / 2, translation[1] - height / 2)
    angle = quaternion_yaw(Quaternion(annotation['rotation'])) / np.pi * 180
    clr = 'green'
    box = Rectangle(xy, width, height, angle = angle, rotation_point = 'center', color = clr, alpha = 0.5)
    ax.add_patch(box)
    ax.set_xlim(translation[0] - 30, translation[0] + 30)
    ax.set_ylim(translation[1] - 30, translation[1] + 30)
def plot_lane(lanes_vertices, lane_token, closest_idx, ax):
    lane = lanes_vertices[lane_token]
    x = []
    y = []
    for node in lane:
        x.append(node.coors[0])
        y.append(node.coors[1])
    for i in range(len(x) - 1):
        ax.arrow(x[i], y[i], x[i+1] - x[i], y[i+1] - y[i], length_includes_head = True, head_width = 0.3, color = 'blue')
    ax.scatter(x[closest_idx], y[closest_idx])