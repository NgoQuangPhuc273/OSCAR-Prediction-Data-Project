import imdb_extract
import rotten_extract
import awards_extract
import box_office_extract

def run():
    try:
        imdb_extract.main()
        print("Successfully extract, tranform and load imdb movies data!")
        print("")
    except:
        print("An error has occurred in imdb_extract.py")
        print("")
        
    try:
        rotten_extract.main()
        print("Successfully extract, tranform and load rotten tomatoes data!")
        print("")
    except:
        print("An error has occurred in rotten_extract.py")
        print("")

    try:
        box_office_extract.main()
        print("Successfully extract, tranform and load box office data!")
        print("")
    except:
        print("An error has occurred in box_office_extract.py")
        print("")
    
    try:
        awards_extract.main()
        print("Successfully extract, tranform and load awards data!")
        print("")
    except:
        print("An error has occurred in awards_extract.py")
        print("")

run()
