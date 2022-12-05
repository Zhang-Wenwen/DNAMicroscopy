from pydna.dseqrecord import Dseqrecord
from PIL import Image as Image
from PIL import ImageDraw as ImageDraw
import numpy as np
import math as math
from scipy.interpolate import CubicSpline as CubicSpline
from pydna.ladders import GeneRuler_1kb_plus as mwstd
import matplotlib.pyplot as plt


def interpolator(mwstd):  # Generate image
    """docstring. """
    interpolator = CubicSpline(
        [len(fr) for fr in mwstd[::-1]],
        [fr.rf for fr in mwstd[::-1]],
        bc_type="natural",
        extrapolate=False,
    )
    interpolator.mwstd = mwstd
    return interpolator


class gel_pos:
    """docstring. modified from pydna, gel simulation package"""

    def __init__(self,
                 samples=None,
                 gel_length=600,
                 margin=10,  # 50
                 interpolator=interpolator(mwstd=mwstd),
                 start=None,
                 n_round=1):
        self.max_intensity = 256
        self.lane_width = 10
        self.lanesep = 10
        self.gel_length = gel_length
        self.samples = samples or [interpolator.mwstd]
        self.width = int(margin * 2 + start[0][-1])
        self.lanes = np.zeros((len(samples), gel_length), dtype=int)
        self.image = Image.new("RGB", (self.width, gel_length), "#ddd")
        self.draw = ImageDraw.Draw(self.image)
        self.draw.rectangle((0, 0, (self.width, gel_length)), fill=(0, 0, 0))  # set canvas as black to draw the band
        self.scale = (gel_length - margin) / interpolator(min(interpolator.x))
        self.start = start
        self.interpolator = interpolator
        self.margin = margin
        self.max_spread = 10
        self.n_round = n_round
        self.final_molecule =  np.zeros((len(samples),1), dtype=int)

    def init_status(self):
        for lane_number, _ in enumerate(self.samples):
            self.draw.rectangle((self.start[0][lane_number], self.start[1][lane_number],
                                 self.start[0][lane_number] + self.lane_width, self.start[1][lane_number] + 1),
                                fill=(256, 256, 256))  # intensity is 0 when there is no molecule
        self.image.show()
        self.draw.rectangle((0, 0, (self.width, self.gel_length)), fill=(0, 0, 0))

    def covert_RGB(self):
        for i, lane in enumerate(self.lanes):  # plot the gel results.
            self.max_intensity = np.amax(self.lanes[i])
            if self.max_intensity > 256:
                self.lanes[i] = np.multiply(self.lanes[i], 256)  # convert it to pixel RGB value
                self.lanes[i] = np.divide(self.lanes[i], self.max_intensity)
        return self.lanes

    def draw_gel(self):
        for i, lane in enumerate(self.lanes):
            m_id = np.where(lane != 0)
            lane = np.concatenate((m_id[0].reshape(1, -1), lane[m_id].reshape(1, -1)),
                                  axis=0)  # first row is the index for band while the second row is the intensity
            for y, intensity in enumerate(lane[1]):  # draw gel band for each lane
                y1 = lane[0][y]
                y2 = lane[0][y] + 1  # y1 and y2 are the height and weight of the band on the gel
                self.draw.rectangle((self.start[0][i], y1, self.start[0][i] + self.lane_width, y2),
                                    fill=(
                                        intensity, intensity, intensity))  # intensity is 0 when there is no molecule

        self.image.show()

    def gel_distribution(self, band, peak_centre, lane_number):
        log = math.log(len(band), 10)
        height = (band.m() / (240 * log)) * 1e10  # band.m(): the mass of the DNA molecule in grams.
        if len(band) < 50:
            peak_centre += 50
            self.max_spread *= 4
            self.max_intensity /= 10
        band_spread = self.max_spread / log
        for i in range(self.max_spread, 0, -1):
            y1 = peak_centre - i
            y2 = peak_centre + i
            intensity = (
                    height
                    * math.exp(
                -float(((y1 - peak_centre) ** 2)) / (2 * (band_spread ** 2))
            )
                    * self.max_intensity
            )
            if y2 > self.gel_length:    # collect the molecule numbers that have arrived at terminal
                self.final_molecule[lane_number] = (y2-self.gel_length) * intensity
                y2 = self.gel_length
            for y in range(int(y1), int(y2)):
                self.lanes[lane_number][y] += intensity

    def gel_pos(self):
        for lane_number, lane in enumerate(self.samples):
            for band in lane:
                peak_centre = 0  # init the start point as 0 for each band and increase after each round
                for n_round in range(self.n_round, 0, -1):
                    self.start[1][lane_number] += peak_centre
                    peak_centre = self.interpolator(len(band)) * self.scale + self.start[1][lane_number]
                    self.gel_distribution(band, peak_centre,
                                          lane_number)  # iterate the start point of molecule on x-axis
        plt.plot(np.transpose(self.lanes)), plt.show()
        self.lanes = self.covert_RGB()  # convert the intensity to rgb values for visualization


if __name__ == "__main__":
    init_pos_x = np.tile(range(10, 70, 20), 3)
    init_pos_y = np.repeat(range(10, 70, 20), 3)
    gel = gel_pos([[Dseqrecord("A" * 8000)]] * 9, gel_length=600,
                  start=[init_pos_x, init_pos_y], n_round=5)  # start point is the x and y coordinate.

    # init_pos_x = range(10, 110, 20)
    # init_pos_y = [10] * 5
    # gel = gel_pos([[Dseqrecord("A" * 8000)]] * 5, gel_length=1000,
    #               start=[init_pos_x, init_pos_y], n_round=3)  # start point is the x and y coordinate.

    gel.init_status()  # show the init status of where the gel is placed

    gel.gel_pos()
    #
    gel.draw_gel()
