import pkg_resources


class PackageHelper:
    @staticmethod
    def get_package_version(package_name):
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
