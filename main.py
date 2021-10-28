from model_init import inference
from utils.visualize import put_logo
import cv2, time

if __name__ == "__main__":

    input_image = './car5.jpg'
    im = cv2.imread(input_image)

    start = time.time()
    model = inference(model_weights_address = "./model/model_final.pth")
    model.load()
    print(f"time (loading model): {time.time()-start}")
    
    start = time.time()
    mask = model.predict(im)
    if mask is not None:
        mask = cv2.resize(mask,(im.shape[1],im.shape[0]), interpolation = cv2.INTER_AREA)
        Detection_flag, result = put_logo(im, mask, cv2.imread("logo.jpg"))
        print(f"time end: {time.time()-start}")
        if Detection_flag:
            # result = np.concatenate((im, result), axis = 1)
            cv2.imwrite('output/'+input_image,result)
        else:
            print("Not found!")
    else:
        print("Not found!")
