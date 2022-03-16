from manipulathor_utils.debugger_util import ForkedPdb


def kinect_reshape(frame):
    frame = frame.copy()
    desired_w, desired_h = 180, 320
    # original_size = desired_h
    assert frame.shape[0] == frame.shape[1]
    original_size = frame.shape[0]
    fraction = max(desired_h, desired_w) / original_size
    beginning = original_size / 2 - desired_w / fraction / 2
    end = original_size / 2 + desired_w / fraction / 2
    frame[:int(beginning), :] = 0
    frame[int(end):, :] = 0
    return frame



def intel_reshape(frame):
    frame = frame.copy()
    desired_w, desired_h = 320,180
    assert frame.shape[0] == frame.shape[1]
    original_size = frame.shape[0]
    fraction = max(desired_h, desired_w) / original_size
    beginning = original_size / 2 - desired_h / fraction / 2
    end = original_size / 2 + desired_h / fraction / 2
    frame[:,:int(beginning)] = 0
    frame[:,int(end):] = 0
    return frame
