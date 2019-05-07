import cgan
if __name__ == '__main__':
    model=cgan.cgan(0.00005,0.5,"C:\\Users\\long\\Desktop\\cgan\\model_save","C:\\Users\\long\\Desktop\\cgan\\rz")
    model.start_train(1000000)