import os
import threading
import tkinter as tk
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

# متغير للتحكم في استمرار تشغيل الحلقة
running = True


# دالة لإنهاء البرنامج عند الضغط على زر "إنهاء"
def exit_program():
    global running
    running = False
    control_root.destroy()  # إغلاق نافذة التحكم


# إنشاء نافذة التحكم باستخدام Tkinter
control_root = tk.Tk()
control_root.title("لوحة التحكم")
exit_button = tk.Button(control_root, text="إنهاء", command=exit_program, font=("Arial", 14), padx=20, pady=10)
exit_button.pack(padx=20, pady=20)


# تشغيل نافذة التحكم في مؤشر ترابط منفصل حتى لا تعيق الحلقة الرئيسية
def run_control():
    control_root.mainloop()


control_thread = threading.Thread(target=run_control, daemon=True)
control_thread.start()

# تهيئة الكاميرا والنموذج
cap = cv2.VideoCapture(0)
classifier = Classifier("C:\\Users\\abdol\\Downloads\\Resources\\Resources\\Model\\keras_model.h5",
                        "C:\\Users\\abdol\\Downloads\\Resources\\Resources\\Model\\labels.txt")
imgArrow = cv2.imread("C:\\Users\\abdol\\Downloads\\Resources\\Resources\\arrow.png", cv2.IMREAD_UNCHANGED)
classIDBin = 0

# استيراد صور النفايات
imgWasteList = []
pathFolderWaste = "C:\\Users\\abdol\\Downloads\\Resources\\Resources\\Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# استيراد صور سلات النفايات
imgBinsList = []
pathFolderBins = "C:\\Users\\abdol\\Downloads\\Resources\\Resources\\Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# قاموس ربط التصنيفات بسلات النفايات:
# 0 = Recyclable, 1 = Hazardous, 2 = Food, 3 = Residual
classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

while running:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)  # يعكس الصورة أفقيًا (انعكاس مرآة)

    imgResize = cv2.resize(img, (454, 340))
    imgBackground = cv2.imread("C:\\Users\\abdol\\Downloads\\Resources\\Resources\\background.png")

    predection = classifier.getPrediction(img)
    classID = predection[1]
    print("تصنيف العنصر:", classID)

    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    cv2.imshow("Output", imgBackground)

    # يمكن إضافة فحص لمفتاح معين (مثلاً "q") أيضًا للخروج
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
