import config
import formats


def main():
    bm_3d = formats.read_bm(config.fn_bm)
    aff_3d = formats.bm2aff(bm_3d)
    formats.save_aff(config.fn_aff, aff_3d)


if __name__ == "__main__":
    main()
