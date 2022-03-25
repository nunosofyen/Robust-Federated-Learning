from art.attacks.evasion import FastGradientMethod, DeepFool, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

def generate_adversarial_data(data,label,classifier,method="fgsm"):
    if (method == "fgsm"):
        # epsilon = 0.1  # Maximum perturbation
        adv_crafter = FastGradientMethod(classifier, eps=0.05, norm='inf', batch_size=1)
        adv_data = adv_crafter.generate(x=data)
        return adv_data
    elif (method == "deepfool"):
        # default norm is 2-nor
        adv_crafter = DeepFool(classifier, epsilon=0.02, batch_size=1, max_iter=5, verbose=False)
        adv_data = adv_crafter.generate(x=data,y=label)
        return adv_data
    elif (method == "pgd"):
        # epsilon = 0.1 # Maximum perturbation
        adv_crafter = ProjectedGradientDescent(classifier, eps=0.05, norm='inf', max_iter=5, batch_size=1, verbose=False)
        adv_data = adv_crafter.generate(x=data)
        return adv_data
