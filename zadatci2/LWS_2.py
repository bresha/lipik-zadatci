'''
 Zadatak 2 - implementacija LDWS koji koristi transformaciju perspektive
 
14.12.2021.
'''


# ovdje definirajte dodatne datoteke ako su vam potrebne
import numpy as np
import cv2
import matplotlib.pyplot as plt


# TODO: napisite funkciju koja oznacava 4 tocke na ulaznoj slici i spaja ih pravcima - za provjeru 4 tocke perspektivne transformacije
def plotArea(image, pts):

    return


# TODO: napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(image):
# TODO: pretvorite sliku iz BGR u HLS
    hsl_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    

    # TODO: definirajte granice za bijelu boju te kreirajte masku pomocu funkcije cv2.inRange
    low_white = np.uint8([0, 150, 0])
    high_white = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(hsl_frame, low_white, high_white)
    # TODO: definirajte granice za zutu boju te kreirajte masku pomocu funkcije cv2.inRange
    low_yellow = np.uint8([20, 120, 127])
    high_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hsl_frame, low_yellow, high_yellow)
    # TODO: kombinirajte obje maske pomocu odgovarajuce logicke operacije (bitwise)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # TODO: filtirajte sliku pomocu dobivene maske koristei odgovarajucu logicku operaciju (bitwise)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, yellow_mask, white_mask, mask


# TODO: napisite funkcija koja detektira dva maksimuma u sumi binarne slike po "vertikali"
def getTwoPeaks(binary_img):

    bin_sum = binary_img.sum(axis=0)
    x_right = bin_sum.argmax()
    bin_sum[x_right] = 0
    x_left = bin_sum.argmax()

    return x_left, x_right


# TODO: prikazite voznu traku u ulaznoj slici; ako vozilo prelazi u drugu traku tada iskljucite prikaz i ispisite upozorenje
def showLane(original_img, x_left, x_right, y1, y2, M_inv):

    overlay = original_img.copy()
    contours = np.array([[x_right, y1], [x_right, y2], [x_left, y1], [x_left, y2]])
    cv2.fillPoly(overlay, [contours], color=(0, 255, 100))

    # cv2.warpPerspective(overlay, M_inv, (width, height), None)

    cv2.addWeighted(original_img, 0.35, overlay, 1-0.35, 0.0, original_img)

    # transformed_img = cv2.warpPerspective(original_img, M_inv, (width, height), None)

    return original_img
    
    



pathResults = 'results/'
pathVideos = 'videos/'
videoName  = 'video2.mp4'

# TODO: Otvorite video pomocu cv2.VideoCapture
cap = cv2.VideoCapture(pathVideos + videoName)

# TODO: Spremite sirinu i visinu video okvira u varijable width i height
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TODO: Otvorite prozore za prikaz video signala i ostale rezultate (neka bude tipa cv2.WINDOW_NORMAL)
in_video_name = 'Input image'
cv2.namedWindow(in_video_name, cv2.WINDOW_NORMAL)
filter_video_name = 'Filtered image'
cv2.namedWindow(filter_video_name, cv2.WINDOW_NORMAL)
trans_video_name = 'Transformed image'
cv2.namedWindow(trans_video_name, cv2.WINDOW_NORMAL)
ret_video_name = 'Retruned perspective'
cv2.namedWindow(ret_video_name, cv2.WINDOW_NORMAL)

# TODO: Definirajte 4 tocke u ulaznoj slici u numpy polju src, 4 tocke u izlaznoj slici u dst numpy polju
# Izracunajte matricu perspektivne transformacije (M) i njen inverz (M^-1)
src = np.array([[375, 626], [1043, 626], [792, 460], [607, 460]], dtype=np.float32)
print(src.dtype)
dst = np.array([[320, 720], [960, 720], [960, 0], [320, 0]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

# TODO: ucitajte sliku i pozovite funkciju koja crta 4 tocke na ulaznoj slici i spaja ih pravcima - kako biste bili sigurni u tocke koje se koriste u transfromaciji
# Trebate najprije pohraniti jedan reprezentativni okvir na disk iz danog video signala
_, frame = cap.read()
pts = np.int32(src)
image = cv2.polylines(frame, [pts], True, (0, 255,0), 3)
cv2.imwrite('1.jpg', image)

k = 0

while(True):


    # TODO: Ucitaj video okvir (frame) pomocu metode read, povecaj k za jedan ako je uspjesno ucitan
    ret, frame = cap.read()
    if ret == False:
        print('Video end')
        break
    else:
        k += 1

    

    # TODO: Pozovite funkciju za filtriranje po boji nad ulaznim okvirom
    filtered_frame, *_ , mask = filterByColor(frame)

    # TODO: Transformirajte filtriranu binarnu sliku
    trans_frame = cv2.warpPerspective(mask, M, (width, height), None, cv2.INTER_LINEAR)

    # TODO: Pozovite funkciju koja pronalazi dva maksimuma u "vertikalnoj sumi" transformirane binarne slike
    x_left, x_right = getTwoPeaks(trans_frame)

    # TODO: Pozovite funkciju koja oznacava voznu traku u originalnom video okviru; u slucaju prelaska u drugu ispisuje upozorenje 
    ret_frame = showLane(frame, x_left, x_right, 0, height, M_inv)

    # TODO: Izracunajte vrijeme obrade u fps


    # TODO: Prikazite vrijeme obrade i redni broj okvira u gornjem lijevom cosku ulaznog video okvira


    # TODO: Prikazite okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite
    cv2.imshow(in_video_name, frame)
    cv2.imshow(filter_video_name, filtered_frame)
    cv2.imshow(trans_video_name, trans_frame)
    cv2.imshow(ret_video_name, ret_frame)

    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break



# TODO: Unistite sve prozore i oslobodite objekt koji je kreiran pomocu cv2.VideoCapture

cv2.destroyAllWindows()
cap.release()