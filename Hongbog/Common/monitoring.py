# import tensorflow as tf
# from tensorflow.python.ops import array_ops
# import numpy as np
#
# alpha = 0.25
# gamma = 2
# prob = np.array([[0.7, 0.3], [0.4, 0.6]])
# y = np.array([0, 1])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     zeros = array_ops.zeros_like(prob, dtype=tf.float32)
#     onehot_y = tf.one_hot(indices=y, depth=2)
#     pos_p_sub = array_ops.where(onehot_y >= prob, onehot_y - prob, zeros)
#     neg_p_sub = array_ops.where(onehot_y > zeros, zeros, prob)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prob, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - prob, 1e-8, 1.0))
#
#     print(sess.run([pos_p_sub]))
#     print(sess.run([neg_p_sub]))




# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Read the data into a pandas DataFrame.
# gender_degree_data = pd.read_csv(
#     "http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv")
#
# # These are the "Tableau 20" colors as RGB.
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#
# # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
# for i in range(len(tableau20)):
#     r, g, b = tableau20[i]
#     tableau20[i] = (r / 255., g / 255., b / 255.)
#
# # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
# # exception because of the number of lines being plotted on it.
# # Common sizes: (10, 7.5) and (12, 9)
# plt.figure(figsize=(12, 14))
#
# # Remove the plot frame lines. They are unnecessary chartjunk.
# ax = plt.subplot(111)
# ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
#
# # Ensure that the axis ticks only show up on the bottom and left of the plot.
# # Ticks on the right and top of the plot are generally unnecessary chartjunk.
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
#
# # Limit the range of the plot to only where the data is.
# # Avoid unnecessary whitespace.
# plt.ylim(0, 90)
# plt.xlim(1968, 2014)
#
# # Make sure your axis ticks are large enough to be easily read.
# # You don't want your viewers squinting to read your plot.
# plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
# plt.xticks(fontsize=14)
#
# # Provide tick lines across the plot to help your viewers trace along
# # the axis ticks. Make sure that the lines are light and small so they
# # don't obscure the primary data lines.
# for y in range(10, 91, 10):
#     plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)
#
# # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
# plt.tick_params(axis="both", which="both", bottom="off", top="off",
#                 labelbottom="on", left="off", right="off", labelleft="on")
#
# # Now that the plot is prepared, it's time to actually plot the data!
# # Note that I plotted the majors in order of the highest % in the final year.
# majors = ['Health Professions', 'Public Administration', 'Education', 'Psychology',
#           'Foreign Languages', 'English', 'Communications\nand Journalism',
#           'Art and Performance', 'Biology', 'Agriculture',
#           'Social Sciences and History', 'Business', 'Math and Statistics',
#           'Architecture', 'Physical Sciences', 'Computer Science',
#           'Engineering']
#
# for rank, column in enumerate(majors):
#     # Plot each line separately with its own color, using the Tableau 20
#     # color set in order.
#     plt.plot(gender_degree_data.Year.values,
#              gender_degree_data[column.replace("\n", " ")].values,
#              lw=2.5, color=tableau20[rank])
#
#     # Add a text label to the right end of every line. Most of the code below
#     # is adding specific offsets y position because some labels overlapped.
#     y_pos = gender_degree_data[column.replace("\n", " ")].values[-1] - 0.5
#     if column == "Foreign Languages":
#         y_pos += 0.5
#     elif column == "English":
#         y_pos -= 0.5
#     elif column == "Communications\nand Journalism":
#         y_pos += 0.75
#     elif column == "Art and Performance":
#         y_pos -= 0.25
#     elif column == "Agriculture":
#         y_pos += 1.25
#     elif column == "Social Sciences and History":
#         y_pos += 0.25
#     elif column == "Business":
#         y_pos -= 0.75
#     elif column == "Math and Statistics":
#         y_pos += 0.75
#     elif column == "Architecture":
#         y_pos -= 0.75
#     elif column == "Computer Science":
#         y_pos += 0.75
#     elif column == "Engineering":
#         y_pos -= 0.25
#
#         # Again, make sure that all labels are large enough to be easily read
#     # by the viewer.
#     plt.text(2011.5, y_pos, column, fontsize=14, color=tableau20[rank])
#
# # matplotlib's title() call centers the title on the plot, but not the graph,
# # so I used the text() call to customize where the title goes.
#
# # Make the title big enough so it spans the entire plot, but don't make it
# # so big that it requires two lines to show.
#
# # Note that if the title is descriptive enough, it is unnecessary to include
# # axis labels; they are self-evident, in this plot's case.
# plt.text(1995, 93, "Percentage of Bachelor's degrees conferred to women in the U.S.A."
#                    ", by major (1970-2012)", fontsize=17, ha="center")
#
# # Always include your data source(s) and copyright notice! And for your
# # data sources, tell your viewers exactly where the data came from,
# # preferably with a direct link to the data. Just telling your viewers
# # that you used data from the "U.S. Census Bureau" is completely useless:
# # the U.S. Census Bureau provides all kinds of data, so how are your
# # viewers supposed to know which data set you used?
# plt.text(1966, -8, "Data source: nces.ed.gov/programs/digest/2013menu_tables.asp"
#                    "\nAuthor: Randy Olson (randalolson.com / @randal_olson)"
#                    "\nNote: Some majors are missing because the historical data "
#                    "is not available for them", fontsize=10)
#
# plt.show()
# # Finally, save the figure as a PNG.
# # You can also save it as a PDF, JPEG, etc.
# # Just change the file extension in this call.
# # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
# # plt.savefig("percent-bachelors-degrees-women-usa.png", bbox_inches="tight")

# import numpy as np
# x1 = np.linspace(0.0, 5.0)
# x2 = np.linspace(0.0, 2.0)
#
# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# y2 = np.cos(2 * np.pi * x2)
#
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('A tale of 2 subplots')
# plt.ylabel('Damped oscillation')
#
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('time (s)')
# plt.ylabel('Undamped')
#
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import time
#
# x = []
# y = []
#
# plt.ion()
# fig = plt.figure()
#
#
# cnt = 0
# for phase in np.linspace(0, 10 * np.pi, 100):
#     cnt = cnt + 1
#     x.append(cnt)
#     y.append(np.random.randint(20))
#
#     acc_line, = acc_plot.plot(x, y, 'b-')
#     loss_line, = loss_plot.plot(x, y, 'b-')
#     acc_line.set_ydata(y)
#     loss_line.set_ydata(y)
#     fig.canvas.draw()
#     time.sleep(1)

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque
import time


class Monitoring:
    MAX_X = 100  # width of graph
    MAX_Y = 1000  # height of graph
    cnt = 1
    x = []  # intialize line to horizontal line on 0
    y = []

    def __init__(self):
        f, (ax1, ax2) = plt.subplots(1, 2)
        l1, = ax1.plot([], [], lw=2)
        ax1.set_title('Accuracy')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)

        l2, = ax2.plot([], [], lw=2)
        ax2.set_title('Loss')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)

        ani = anim.FuncAnimation(f, self.update, fargs=(l1, l2), interval=50)
        plt.show()

    def update(self, fn, l1, l2):
        # print(self.x, self.y)
        l1.set_data(self.x, self.y)
        l2.set_data(self.x, self.y)

    def add_data(self, value):
        self.x.append(self.cnt)
        self.y.append(value)
        self.cnt = self.cnt + 1

mon = Monitoring()
print(1)
while True:
    mon.add_data(40)
    print(mon.x, mon.y)
    time.sleep(1)